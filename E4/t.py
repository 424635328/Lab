import os
# 设置 KMP_DUPLICATE_LIB_OK 环境变量以解决 OpenMP 冲突
# This should be set BEFORE importing torch or numpy heavily
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import shutil
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import glob # 用于查找文件路径
import random # 用于随机选择示例
import json # Uncomment if you need to save report_dict as JSON
import traceback # Import traceback for detailed error printing

# --- Configuration Parameters ---
# Please set this path to the directory containing the colonoscopy_dataset folder
DATASET_BASE_DIR = "./" # Example: "./" if colonoscopy_dataset is in the same directory as the script
DATASET_ROOT_NAME = "colonoscopy_dataset" # Name of the dataset root folder
LABELS_FILE = os.path.join(DATASET_BASE_DIR, DATASET_ROOT_NAME, "HumanEvaluation.xlsx") # Path to the labels file (modify as needed)
OUTPUT_DIR = "./experiment_outputs" # Output directory for results and images

# Dataset Classes - Must match the standardized text in the GROUND TRUTH column of HumanEvaluation.xlsx (Capitalized)
CLASSES = ['Hyperplasic', 'Serrated', 'Adenoma']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

# Actual column names and header row (0-indexed) in HumanEvaluation.xlsx
EXCEL_CASE_ID_COLUMN = 'LESION'
EXCEL_LABEL_COLUMN = 'GROUND TRUTH'
EXCEL_HEADER_ROW = 2 # Excel header is on the 3rd row (index 2)

# Video Frame Extraction Strategy
FRAMES_PER_VIDEO = 30 # Number of frames to uniformly extract from each video

# Image Preprocessing Parameters
IMG_SIZE = (224, 224) # Model input image size (e.g., 224x224 for ResNet)
BATCH_SIZE = 32
# NUM_WORKERS = 0 # Number of processes for data loading (adjust based on your system, 0 for no multiprocessing)
# Keeping NUM_WORKERS = 0 as per previous runs to simplify debugging DataLoader issues
NUM_WORKERS = 0


# Model Training Parameters
MODEL_NAME = 'resnet18' # The pretrained model architecture: resnet18, resnet50 etc.
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
TRAIN_VALID_SPLIT_RATIO = 0.8 # Training set + Validation set percentage of total frames
VALID_TEST_SPLIT_RATIO = 0.5 # Validation and Test sets each take this percentage of the remaining frames

# --- Model Saving Path ---
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_model_final.pth") # Path to save the final trained model state_dict
# --------------------------

# Other
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Functions ---

def create_output_dirs():
    """Create output directory structure, clearing previous run data."""
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing previous output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR) # Clear all previous outputs for a clean run
    os.makedirs(os.path.join(OUTPUT_DIR, 'data_examples'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'data_augmentation_examples'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'extracted_frames'), exist_ok=True) # Directory for saving extracted frames
    os.makedirs(os.path.join(OUTPUT_DIR, 'grad_cam'), exist_ok=True) # Directory for Grad-CAM visualizations
    os.makedirs(os.path.join(OUTPUT_DIR, 'evaluation_plots'), exist_ok=True) # Directory for evaluation plots
    print(f"Created new output directory: {OUTPUT_DIR}")


def extract_frames_from_video(video_path, num_frames=FRAMES_PER_VIDEO):
    """Extract specified number of frames uniformly from a video."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # print(f"Error: Could not open video {video_path}") # Optional: too verbose if many missing videos
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # print(f"Warning: Video {video_path} has 0 frames or could not read frame count.")
        cap.release() # Release resource
        return frames

    # Select frame indices uniformly
    if total_frames < num_frames:
        # If fewer frames than requested, extract all
        indices = list(range(total_frames))
    else:
        # Ensure indices are within valid range [0, total_frames - 1]
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        indices = np.clip(indices, 0, total_frames - 1) # Just in case linspace edges are off by a tiny bit

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert OpenCV BGR format to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb) # Save as numpy array
        else:
            print(f"Warning: Could not read frame {i} from {video_path}")

    cap.release()
    return frames

# --- Dataset Class ---
# This dataset class loads the *saved* extracted frames
class ColonoscopyFrameDataset(Dataset):
    def __init__(self, image_info_list, transform=None):
        # image_info_list: list of dictionaries, e.g.,
        # [{'img_path': 'path/to/frame.jpg', 'label': 0, 'label_str': 'Hyperplasic', 'case_id': 'adenoma_02', 'mode': 'WL', 'frame_idx': 0}, ...]
        self.image_info_list = image_info_list
        self.transform = transform

    def __len__(self):
        return len(self.image_info_list)

    def __getitem__(self, idx):
        img_info = self.image_info_list[idx]
        img_path = img_info['img_path']
        label = img_info['label']

        try:
            # Use PIL to load image, compatible with torchvision transform
            image = Image.open(img_path).convert('RGB') # Ensure RGB format

            if self.transform:
                image = self.transform(image)

            # Return image tensor, label (int), and the info dictionary
            return image, label, img_info

        except Exception as e:
            # Handle errors during image loading or transformation
            print(f"Error processing image {img_path} (index {idx}): {e}")
            # Return None for image and label, but return the img_info dictionary
            # This allows the custom collate_fn to filter out the bad image
            # but still keep the metadata for potential debugging or tracking.
            # Returning label -1 as a placeholder for a failed sample.
            return None, -1, img_info


# --- Custom Collate Function ---
# This function is used by DataLoader to combine individual samples into batches.
# It handles cases where __getitem__ returns None for the image.
def custom_collate_fn(batch):
    # 'batch' is a list of tuples from Dataset.__getitem__, e.g.,
    # [(img_tensor_1, label_1, info_dict_1), (img_tensor_2, label_2, info_dict_2), ...]
    # Some img_tensor might be None if loading/transform failed.

    # Filter out samples where image loading/transform failed (image is None)
    # Keep only the items where the first element (image) is not None.
    # Ensure the item is a tuple of length 3 first for safety.
    batch = [item for item in batch if isinstance(item, (list, tuple)) and len(item) == 3 and item[0] is not None]

    # If the batch is empty after filtering (e.g., all samples failed), return empty tensors and lists.
    if not batch:
        # Return empty tensors with expected shapes and empty info list
        # The shape [0, 3, IMG_SIZE[0], IMG_SIZE[1]] indicates an empty batch of images
        return torch.empty(0, 3, IMG_SIZE[0], IMG_SIZE[1]), torch.empty(0, dtype=torch.long), []

    # Separate the components of the valid samples
    images = [item[0] for item in batch]   # List of image tensors
    labels = [item[1] for item in batch]   # List of label integers
    infos = [item[2] for item in batch]    # List of img_info dictionaries

    # Stack the valid image tensors and label integers
    images_tensor = torch.stack(images, 0) # Stack list of tensors into a batch tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long) # Convert list of ints to a label tensor

    # Return the stacked tensors and the corresponding list of info dictionaries
    # The length of images_tensor, labels_tensor, and infos will now match
    collated_batch = (images_tensor, labels_tensor, infos)

    return collated_batch


# --- Model Building ---
def build_model(model_name, num_classes):
    """Load pretrained model architecture and modify classifier."""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True) # Use pretrained weights for transfer learning
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) # Replace the classifier
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True) # Use pretrained weights for transfer learning
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) # Replace the classifier
    # Add more models here if needed
    else:
        raise ValueError(f"Model architecture '{model_name}' is not supported in build_model.")

    return model

# --- Training and Validation Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE, model_save_path=None):
    """Train the model and validate periodically."""
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    model.to(device) # Move model to GPU if available

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        processed_samples_train = 0

        # Iterate over data
        # DataLoader returns inputs, labels, img_info (info is ignored in train/val)
        # Using custom_collate_fn, inputs and labels only contain valid samples
        for inputs, labels, _ in train_loader:
            # Skip batch if no valid images were loaded
            if inputs.size(0) == 0:
                 continue

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_samples_train += inputs.size(0) # Count actual samples processed


        # Calculate epoch loss and accuracy based on processed samples
        epoch_loss = running_loss / processed_samples_train if processed_samples_train > 0 else 0.0
        epoch_acc = running_corrects.double() / processed_samples_train if processed_samples_train > 0 else 0.0

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc)


        print(f'Epoch {epoch+1}/{num_epochs} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ({processed_samples_train} samples)')

        # --- Validation Phase ---
        model.eval() # Set model to evaluate mode
        running_loss_val = 0.0
        running_corrects_val = 0
        processed_samples_val = 0

        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs_val, labels_val, _ in val_loader:
                 # Skip batch if no valid images were loaded
                if inputs_val.size(0) == 0:
                     continue

                inputs_val = inputs_val.to(device)
                labels_val = labels_val.to(device)

                outputs_val = model(inputs_val)
                _, preds_val = torch.max(outputs_val, 1)
                loss_val = criterion(outputs_val, labels_val)

                running_loss_val += loss_val.item() * inputs_val.size(0)
                running_corrects_val += torch.sum(preds_val == labels_val.data)
                processed_samples_val += inputs_val.size(0)


        epoch_loss_val = running_loss_val / processed_samples_val if processed_samples_val > 0 else 0.0
        epoch_acc_val = running_corrects_val.double() / processed_samples_val if processed_samples_val > 0 else 0.0

        val_losses.append(epoch_loss_val)
        val_accs.append(epoch_acc_val.item() if isinstance(epoch_acc_val, torch.Tensor) else epoch_acc_val)


        print(f'Epoch {epoch+1}/{num_epochs} Val Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f} ({processed_samples_val} samples)')

        # --- Optional: Save checkpoint based on validation accuracy ---
        # You could add logic here to save the model if it achieves the best validation accuracy seen so far.
        # Example:
        # if epoch_acc_val > best_val_acc:
        #    best_val_acc = epoch_acc_val
        #    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_checkpoint.pth'))
        #    print("-> Saved best model checkpoint")
        # -------------------------------------------------------------


    print("Training complete.")

    # --- Save the final trained model state_dict ---
    if model_save_path:
        print(f"Saving final trained model state_dict to {model_save_path}...")
        try:
            torch.save(model.state_dict(), model_save_path)
            print("Final model state_dict saved successfully.")
        except Exception as e:
            print(f"Error saving model state_dict to {model_save_path}: {e}")
    # ------------------------------------------------


    return model, train_losses, train_accs, val_losses, val_accs


# --- Evaluation Function ---
# Modified to correctly collect info for processed samples using custom_collate_fn
def evaluate_model(model, test_loader, device=DEVICE):
    """Evaluate model on the test set and calculate metrics."""
    model.eval() # Set model to evaluation mode
    all_labels = [] # Ground truth labels for successfully processed samples
    all_preds = [] # Predicted labels for successfully processed samples
    all_probs = [] # Predicted probabilities for successfully processed samples
    # Collect test set image information *during* evaluation loop, for successfully processed samples
    test_image_info_collected = []

    print("Starting evaluation on test set...")
    with torch.no_grad(): # Disable gradient calculation
        # test_loader now uses custom_collate_fn, returns (images_tensor, labels_tensor, infos_list_valid)
        # where infos_list_valid contains infos ONLY for valid images in that batch
        for batch_idx, (inputs, labels, img_infos_batch_valid) in enumerate(test_loader):

            # Skip batch if no valid images were loaded
            if inputs.size(0) == 0:
                 # print(f"Warning: Skipping evaluation batch {batch_idx} with 0 valid images.")
                 continue

            inputs = inputs.to(device)
            labels = labels.to(device) # These labels correspond to 'inputs'

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Extend lists only with data from successfully processed samples in this batch
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Extend the info list with infos ONLY for the successfully processed samples in this batch
            test_image_info_collected.extend(img_infos_batch_valid)

    # After the loop, the lengths of all_labels, all_preds, all_probs, and test_image_info_collected
    # should match the total number of samples successfully loaded and evaluated.
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # This check should now ideally show 0 mismatch if the custom_collate_fn works correctly
    # and all samples in the DataLoader output are processed.
    if len(all_labels) != len(test_image_info_collected):
        # This indicates a severe issue if it still happens after custom collate_fn
        print(f"\nSEVERE ERROR: Post-evaluation mismatch between number of evaluated samples ({len(all_labels)}) and collected info items ({len(test_image_info_collected)}).")
        print("This indicates a fundamental issue in data loading/collection logic. Grad-CAM will likely fail or be unreliable.")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback here if needed

    print("\n--- Evaluation Results ---")

    # Wrap evaluation metrics calculation and plotting in a try-except block
    # to prevent crash if e.g., a class is missing in the test set results.
    cm = np.array([]) # Initialize cm and report_dict before try block
    report_dict = {}
    try:
        # Classification Report (Accuracy, Precision, Recall, F1-score)
        # Ensure there are samples to evaluate
        if len(all_labels) > 0:
            report_str = classification_report(all_labels, all_preds, target_names=CLASSES)
            print(report_str)
            report_dict = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)

            # Save classification report to file
            report_file_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
            with open(report_file_path, 'w') as f:
                f.write("Classification Report:\n")
                f.write(report_str)
            print(f"Classification report saved to {report_file_path}")

            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            cm_path = os.path.join(OUTPUT_DIR, 'evaluation_plots', 'confusion_matrix.png')
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
            plt.close()

            # ROC Curve and AUC
            plt.figure(figsize=(10, 8))
            # Check if there's more than one class present in true labels for ROC curve
            if len(np.unique(all_labels)) > 1:
                for i in range(len(CLASSES)):
                    # roc_curve needs (true_binary_labels, predicted_probabilities)
                    # true_binary_labels: Set current class to 1, others to 0
                    true_binary = (all_labels == i).astype(int)
                    # Handle cases where a class might not be present in true labels or has only one sample
                    # roc_curve requires at least two samples with different true labels (0 and 1)
                    unique_true_binary = np.unique(true_binary)
                    if len(unique_true_binary) > 1:
                        fpr, tpr, _ = roc_curve(true_binary, all_probs[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, label=f'{CLASSES[i]} (AUC = {roc_auc:.2f})')
                    else:
                         # print(f"Warning: Class {CLASSES[i]} has only one true label type ({unique_true_binary[0]}) in test set results, cannot plot ROC curve.")
                         pass # Keep printing cleaner if not critical

                plt.plot([0, 1], [0, 1], 'k--') # Random guess line
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                roc_path = os.path.join(OUTPUT_DIR, 'evaluation_plots', 'roc_curve.png')
                plt.savefig(roc_path)
                print(f"ROC curve saved to {roc_path}")
                plt.close()
            else:
                print("Warning: Only one class present in test set true labels. Cannot plot ROC curve.")


            # Precision-Recall Curve
            plt.figure(figsize=(10, 8))
            if len(np.unique(all_labels)) > 1: # PR curve also typically requires more than one class
                for i in range(len(CLASSES)):
                    # PrecisionRecallDisplay needs (true_binary_labels, predicted_probabilities_of_positive_class)
                    true_binary = (all_labels == i).astype(int)
                     # Handle cases where a class might not be present or has only one sample
                    unique_true_binary = np.unique(true_binary)
                    if len(unique_true_binary) > 1:
                        # Use PrecisionRecallDisplay.from_predictions
                        display = PrecisionRecallDisplay.from_predictions(true_binary, all_probs[:, i], name=CLASSES[i])
                        display.plot(ax=plt.gca(), name=CLASSES[i]) # Plot onto the current axes
                    else:
                         # print(f"Warning: Class {CLASSES[i]} has only one true label type ({unique_true_binary[0]}) in test set results, cannot plot Precision-Recall curve.")
                         pass


                plt.title('Precision-Recall Curve')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.grid(True)
                pr_curve_path = os.path.join(OUTPUT_DIR, 'evaluation_plots', 'precision_recall_curve.png')
                plt.savefig(pr_curve_path)
                print(f"Precision-Recall curve saved to {pr_curve_path}")
                plt.close()
            else:
                 print("Warning: Only one class present in test set true labels. Cannot plot Precision-Recall curve.")

        else:
             print("No samples were successfully evaluated. Skipping evaluation metrics and plots.")


    except Exception as e:
        # If any error occurs during calculation or plotting
        print(f"\nError encountered during evaluation metrics calculation or plotting: {e}")
        import traceback
        traceback.print_exc() # Print detailed error information
        # cm and report_dict are already initialized with default values
        print("Evaluation metrics calculation failed. Returning available data.")


    # Return the collected data for successfully processed samples
    return cm, report_dict, all_labels, all_preds, all_probs, test_image_info_collected

# --- Grad-CAM Visualization (Simple Implementation) ---
# Needs target_layer_name adjusted based on specific model architecture
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self.model.eval() # Ensure model is in evaluation mode
        # Hooks will be registered before use in generate_heatmap

    def hook_layers(self):
        # Find the target layer by name and register forward/backward hooks
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                # Register hooks
                # Ensure hooks are removed before registering new ones if method is called multiple times
                self.remove_hooks()
                self.forward_hook_handle = target_layer.register_forward_hook(self._forward_hook)
                self.backward_hook_handle = target_layer.register_backward_hook(self._backward_hook)
                # print(f"Hooks registered for layer: {self.target_layer_name}") # Optional
                break

        if target_layer is None:
            raise RuntimeError(f"Target layer '{self.target_layer_name}' not found in model.")

    def _forward_hook(self, module, input, output):
        # output is the output tensor of the layer
        self.activations = output.cpu().data # Save activations

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple (gradient w.r.t output, ...), we need the gradient w.r.t. the output
        self.gradients = grad_output[0].cpu().data # Save gradients flowing into the layer

    def remove_hooks(self):
        # Remember to remove hooks after visualization to avoid memory leaks or interfering with other operations
        if hasattr(self, 'forward_hook_handle') and self.forward_hook_handle is not None:
             self.forward_hook_handle.remove()
             self.forward_hook_handle = None # Clear handle after removing
        if hasattr(self, 'backward_hook_handle') and self.backward_hook_handle is not None:
             self.backward_hook_handle.remove()
             self.backward_hook_handle = None # Clear handle after removing
        # print("Grad-CAM hooks removed.") # Optional


    def generate_heatmap(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap for a single input image.
        input_image: torch.Tensor of shape (1, C, H, W), already normalized and on device.
        target_class: int, the class to generate heatmap for. If None, use model's prediction.
        Returns: numpy array (H_resized, W_resized) heatmap scaled to [0, 1]
                 or None if heatmap generation failed.
        """
        # Ensure model is on the correct device
        self.model.to(input_image.device)
        self.model.eval() # Ensure model is in evaluation mode

        # Register hooks before forward pass
        try:
             self.hook_layers()
        except RuntimeError as e:
             print(f"Error hooking layers for Grad-CAM: {e}")
             return None # Cannot proceed without hooks


        # Copy image to keep original tensor detached from graph
        # Ensure input_image has requires_grad=True for backward pass
        # This requires the input to the model have requires_grad=True.
        # For a single image inference, we can clone and set requires_grad=True.
        if not input_image.requires_grad:
             input_image_for_grad = input_image.clone().requires_grad_(True)
        else:
             input_image_for_grad = input_image.clone() # If already requires_grad, just clone


        try:
            # Forward pass
            output = self.model(input_image_for_grad)

            # Get predicted class if target_class is not specified
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Ensure target_class is valid for the model output dimension
            if target_class < 0 or target_class >= output.size(1):
                 print(f"Warning: Invalid target_class {target_class} for output dimension {output.size(1)}. Using predicted class.")
                 target_class = output.argmax(dim=1).item() # Fallback to predicted class


            # Zero gradients for the model parameters
            # NOTE: We zero the gradients *before* backward pass for this specific sample
            self.model.zero_grad()

            # Compute gradient of the target class score with respect to model output
            # Select the score for the target class from the output
            target_score = output[0, target_class]

            # Perform backward pass to get gradients
            # retain_graph=True is needed if you need the graph for further backward passes (e.g., for other samples in a batch, or multiple Grad-CAMs).
            # Here, we do one sample at a time, but keeping retain_graph=True is safer if the context changes.
            # However, if not needed, setting it to False saves memory. Let's keep it False for single image processing.
            target_score.backward(retain_graph=False)


            # Get gradients and activations from saved hook data
            # Ensure activations and gradients were successfully saved by the hooks
            if self.gradients is None or self.activations is None:
                 print("Error: Gradients or Activations were not captured by hooks. Check target_layer_name.")
                 return None # Return None if data wasn't captured

            gradients = self.gradients # shape: (N, C_layer, H_layer, W_layer), N=1 for batch size 1
            activations = self.activations # shape: (N, C_layer, H_layer, W_layer), N=1

            # Remove the batch dimension (N=1)
            gradients = gradients.squeeze(0) # shape: (C_layer, H_layer, W_layer)
            activations = activations.squeeze(0) # shape: (C_layer, H_layer, W_layer)


            # Compute weights: Global Average Pooling of gradients over spatial dimensions
            # Mean over height and width for each channel
            weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # shape: (C_layer, 1, 1)

            # Weighted sum of activations and apply ReLU
            # Expand weights to match spatial dimensions of activations for element-wise multiplication
            weighted_activations = weights * activations # Broadcast multiplication
            heatmap = torch.sum(weighted_activations, dim=0) # Sum over channels to get shape: (H_layer, W_layer)
            heatmap = torch.relu(heatmap) # Apply ReLU to the heatmap

            # Normalize heatmap to [0, 1]
            heatmap_max = torch.max(heatmap)
            if heatmap_max > 0:
                 heatmap = heatmap / heatmap_max
            else:
                 # Handle case where heatmap is all zeros (e.g., if gradients were zero or ReLU removed everything)
                 heatmap = torch.zeros_like(heatmap)


            # Resize heatmap to the target image size (model input size IMG_SIZE)
            heatmap_np = heatmap.detach().cpu().numpy() # Move to CPU and convert to numpy
            heatmap_resized = cv2.resize(heatmap_np, IMG_SIZE) # Resize to model input size

            return heatmap_resized

        except Exception as e:
            print(f"Error generating Grad-CAM heatmap: {e}")
            import traceback
            traceback.print_exc()
            return None # Return None if heatmap generation failed

        finally:
            # Ensure hooks are removed even if an error occurred during processing
            self.remove_hooks()
            # Clear saved gradients and activations to prevent issues in the next call
            self.gradients = None
            self.activations = None


    def save_gradcam_image(self, original_frame_path_saved, heatmap, predicted_class_str, true_class_str, img_info, save_path):
        """
        Overlay heatmap on the original image and save.
        original_frame_path_saved: Path to the saved extracted frame image (used to load original pixels).
        heatmap: numpy array (H, W) heatmap scaled to [0, 1] and resized to IMG_SIZE.
        predicted_class_str: string of the predicted class.
        true_class_str: string of the true class.
        img_info: dict containing frame info (case_id, mode, frame_idx).
        save_path: Path to save the final superimposed image.
        Returns True if successful, False otherwise.
        """
        if heatmap is None:
             print(f"Error: Cannot save Grad-CAM image to {save_path} because heatmap generation failed.")
             return False

        try:
            # Load the original saved extracted frame image
            original_image = cv2.imread(original_frame_path_saved)
            if original_image is None:
                print(f"Error: Could not load image for Grad-CAM overlay: {original_frame_path_saved}")
                return False

            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # Convert to RGB

            # Resize original image to IMG_SIZE for overlay
            original_image_resized = cv2.resize(original_image, IMG_SIZE)

            # Convert heatmap to a pseudo-colored image (Jet colormap)
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Overlay heatmap on the resized original image using alpha blending
            # Result = alpha * src1 + beta * src2 + gamma
            # Adjust alpha/beta values for desired transparency
            superimposed_img = cv2.addWeighted(original_image_resized, 0.7, heatmap_colored, 0.3, 0)
            superimposed_img = np.uint8(superimposed_img) # Ensure output is uint8


            # Add text labels for True/Predicted Class and Image Info
            label_text = f"True: {true_class_str}, Pred: {predicted_class_str}"
            info_text = f"Case: {img_info['case_id']}, Mode: {img_info['mode']}, Frame: {img_info['frame_idx']:03d}"
            text_color = (255, 255, 255) # White color
            # Add text with a simple black shadow for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            shadow_thickness = thickness + 1
            text_offset_y = 25 # Vertical spacing between lines
            text_x = 10

            # Text line 1: True/Pred
            cv2.putText(superimposed_img, label_text, (text_x, text_offset_y), font, font_scale, (0, 0, 0), shadow_thickness, cv2.LINE_AA) # Shadow
            cv2.putText(superimposed_img, label_text, (text_x, text_offset_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Text line 2: Info
            cv2.putText(superimposed_img, info_text, (text_x, text_offset_y * 2), font, font_scale, (0, 0, 0), shadow_thickness, cv2.LINE_AA) # Shadow
            cv2.putText(superimposed_img, info_text, (text_x, text_offset_y * 2), font, font_scale, text_color, thickness, cv2.LINE_AA)


            # Save the image
            superimposed_img_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR) # Convert back to BGR for saving with cv2.imwrite
            cv2.imwrite(save_path, superimposed_img_bgr)
            return True

        except Exception as e:
            print(f"Error saving Grad-CAM image {save_path}: {e}")
            import traceback
            traceback.print_exc()
            return False


# --- Main Execution Function ---

def main():
    # Set random seeds for reproducibility (partial)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED) # Seed for random module as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True # Ensures deterministic convolution algorithms
        torch.backends.cudnn.benchmark = False # Disables cuDNN benchmark for deterministic behavior


    # 1. Create output directories
    create_output_dirs()

    # 2. Read label information and build complete video file list
    print("Reading labels from Excel file and building video list...")
    video_info_list = [] # List to store dictionaries like {'video_path': '...', 'label': 0, 'label_str': '...', 'case_id': '...', 'mode': '...'}
    try:
        # Read Excel file, specify header row (0-indexed)
        df_labels = pd.read_excel(LABELS_FILE, header=EXCEL_HEADER_ROW)

        # Use actual column names from HumanEvaluation.xlsx
        case_id_col = EXCEL_CASE_ID_COLUMN
        label_col = EXCEL_LABEL_COLUMN

        if case_id_col not in df_labels.columns or label_col not in df_labels.columns:
             print(f"Error: Expected columns '{case_id_col}' and '{label_col}' not found in {LABELS_FILE}.")
             print(f"Columns found: {df_labels.columns.tolist()}") # Print all found column names for debugging
             raise KeyError(f"Expected columns '{case_id_col}' and '{label_col}' not found.")

        # Handle label case inconsistency and map labels to numeric indices
        label_mapping_lower_to_capital = {
            'hyperplasic': 'Hyperplasic',
            'serrated': 'Serrated',
            'adenoma': 'Adenoma'
        }
        # Convert label column to standardized capitalized format
        # Use .str.lower() to ensure handling of mixed case in Excel
        df_labels['standard_label'] = df_labels[label_col].astype(str).str.lower().map(label_mapping_lower_to_capital) # Ensure column is string type before lower()

        # Filter out entries with labels not in our CLASSES list (if any) or where mapping failed (resulted in NaN)
        df_labels = df_labels[df_labels['standard_label'].isin(CLASSES)].reset_index(drop=True)

        if df_labels.empty:
            print("Error: No entries found in the labels file matching the specified CLASSES after filtering.")
            return # Script terminates

        # Add numeric label column
        df_labels['label'] = df_labels['standard_label'].map(CLASS_TO_IDX)
        # ------------------------------------------------------------------

        # --- Build video file list based on processed DataFrame ---
        # Ensure DATASET_BASE_DIR points to the directory containing colonoscopy_dataset
        dataset_full_path = os.path.join(DATASET_BASE_DIR, DATASET_ROOT_NAME)

        for index, row in df_labels.iterrows():
            case_id = str(row[case_id_col]) # Ensure case ID is string format
            label_idx = row['label']       # Numeric label
            label_str = row['standard_label'] # Standardized label string

            # Construct case directory path
            # Assuming structure is DATASET_BASE_DIR/colonoscopy_dataset/Class/CaseID/videos/Mode.mp4
            # Class directory name should match label_str (Hyperplasic, Serrated, Adenoma)
            class_dir = label_str

            case_dir_path = os.path.join(dataset_full_path, class_dir, case_id)
            videos_dir_path = os.path.join(case_dir_path, 'videos')

            # Check if videos directory exists
            if not os.path.exists(videos_dir_path):
                 # print(f"Warning: Videos directory not found for case {case_id} ({label_str}) at {videos_dir_path}. Skipping.")
                 continue # Skip cases where video directory is not found

            # Look for WL and NBI video files
            wl_video_path = os.path.join(videos_dir_path, 'WL.mp4')
            nbi_video_path = os.path.join(videos_dir_path, 'NBI.mp4')

            if os.path.exists(wl_video_path):
                 video_info_list.append({'video_path': wl_video_path, 'label': label_idx, 'label_str': label_str, 'case_id': case_id, 'mode': 'WL'})
            else:
                 # Not all cases might have both WL or NBI videos. Skip if a mode file doesn't exist.
                 # print(f"Info: WL video not found for case {case_id} at {wl_video_path}. Skipping.")
                 pass # Don't print warning

            if os.path.exists(nbi_video_path):
                 video_info_list.append({'video_path': nbi_video_path, 'label': label_idx, 'label_str': label_str, 'case_id': case_id, 'mode': 'NBI'})
            else:
                  # print(f"Info: NBI video not found for case {case_id} at {nbi_video_path}. Skipping.")
                 pass # Don't print warning


        print(f"Found {len(video_info_list)} individual video files across cases based on '{case_id_col}' and '{label_col}' columns.")

    except FileNotFoundError:
        print(f"Error: Labels file not found at {LABELS_FILE}")
        return # Script terminates
    except KeyError as e:
        print(f"Error: An expected column or label value was not found. {e}.")
        print(f"Please check column names ('{case_id_col}', '{label_col}') and label spellings ('hyperplasic', 'serrated', 'adenoma') in {LABELS_FILE}.")
        print(f"Also ensure the CLASSES list in the script matches the expected labels ({CLASSES}).")
        return # Script terminates
    except Exception as e:
        print(f"An unexpected error occurred while reading the labels file or building video list: {e}")
        import traceback
        traceback.print_exc() # Print detailed error information
        return # Script terminates

    if not video_info_list:
        print("No valid video files found based on the labels file and directory structure. Please check file paths and labels.")
        return # Script terminates

    # --- 图片输出 1: Data Examples ---
    print("\nSaving data examples...")
    example_count_per_class_mode = 3 # Save this many examples per class/mode combination
    examples_saved = {cls: {'WL': 0, 'NBI': 0} for cls in CLASSES}
    # Shuffle the video_info_list to get random examples
    random.shuffle(video_info_list)

    # Iterate through videos to find examples
    for video_info in video_info_list:
        v_path = video_info['video_path']
        v_label_str = video_info['label_str']
        v_mode = video_info['mode']
        v_case_id = video_info['case_id']

        # Check if we need more examples for this class/mode
        if v_label_str in CLASSES and v_mode in ['WL', 'NBI'] and examples_saved[v_label_str][v_mode] < example_count_per_class_mode:
            frames = extract_frames_from_video(v_path, num_frames=1) # Extract one frame as example
            if frames:
                frame = frames[0]
                frame_pil = Image.fromarray(frame)
                # Construct save path: Class_CaseID_Mode_frame0.jpg
                save_name = f"{v_label_str}_{v_case_id}_{v_mode}_frame0.jpg"
                save_path = os.path.join(OUTPUT_DIR, 'data_examples', save_name)
                try:
                    frame_pil.save(save_path)
                    examples_saved[v_label_str][v_mode] += 1
                    # print(f"Saved example: {save_path}") # Optional: too verbose
                except Exception as e:
                    print(f"Warning: Could not save example image {save_path}. Error: {e}")


        # Stop once enough examples for all target class/mode combinations are saved
        stop_saving_examples = True
        for cls in CLASSES:
            if examples_saved[cls]['WL'] < example_count_per_class_mode or examples_saved[cls]['NBI'] < example_count_per_class_mode:
                stop_saving_examples = False
                break
        if stop_saving_examples:
            print("Finished saving data examples.")
            break # Exit the loop over videos

    # --- 图片输出 2: Video File Distribution ---
    print("\nSaving video file distribution plot...")
    df_video_info = pd.DataFrame(video_info_list)
    if 'label_str' in df_video_info.columns and not df_video_info.empty:
        video_counts = df_video_info.groupby(['label_str', 'mode']).size().unstack(fill_value=0)
        video_counts = video_counts.reindex(CLASSES, fill_value=0)
        valid_modes = [mode for mode in ['WL', 'NBI'] if mode in video_counts.columns]
        video_counts = video_counts[valid_modes] if valid_modes else pd.DataFrame(index=CLASSES)

        if not video_counts.empty and not video_counts.sum().sum() == 0:
            ax = video_counts.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis')
            plt.title('Number of Video Files per Class and Mode')
            plt.xlabel('Class')
            plt.ylabel('Number of Video Files')
            plt.xticks(rotation=0)
            plt.legend(title='Mode')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            distribution_plot_path = os.path.join(OUTPUT_DIR, 'video_file_distribution.png')
            plt.savefig(distribution_plot_path)
            print(f"Video file distribution plot saved to {distribution_plot_path}")
            plt.close()
        else:
             print("No video files with valid labels found to plot distribution.")
    else:
        print("Could not generate video distribution plot: No video information collected.")


    # 3. Extract video frames and collect all image information
    print("\nExtracting frames from videos...")
    all_frame_info_list = [] # List of dictionaries for each extracted frame
    frames_save_dir = os.path.join(OUTPUT_DIR, 'extracted_frames') # Directory for saving extracted frames

    # Use enumerate for progress tracking
    for i, video_info in enumerate(video_info_list):
        v_path = video_info['video_path']
        v_label_idx = video_info['label']
        v_label_str = video_info['label_str']
        v_case_id = video_info['case_id']
        v_mode = video_info['mode']

        # print(f"Processing video {i+1}/{len(video_info_list)}: {os.path.basename(v_path)}") # Optional: too verbose

        frames = extract_frames_from_video(v_path, num_frames=FRAMES_PER_VIDEO)

        if not frames:
            # print(f"No frames extracted from {os.path.basename(v_path)}") # Optional: too verbose
            continue # Skip if no frames were extracted

        # Construct subdirectory for saving frames: Class/CaseID_Mode
        frame_sub_dir = os.path.join(v_label_str, f"{v_case_id}_{v_mode}")
        os.makedirs(os.path.join(frames_save_dir, frame_sub_dir), exist_ok=True)

        for j, frame in enumerate(frames):
            # Save frame as image file
            frame_filename = f"frame_{j:03d}.jpg" # Format frame number, e.g., frame_000.jpg
            frame_path_saved = os.path.join(frames_save_dir, frame_sub_dir, frame_filename)
            # cv2.imwrite needs BGR format
            try:
                cv2.imwrite(frame_path_saved, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                all_frame_info_list.append({
                    'img_path': frame_path_saved,
                    'label': v_label_idx,
                    'label_str': v_label_str,
                    'case_id': v_case_id,
                    'mode': v_mode,
                    'frame_idx': j # Record frame index within the video
                })
            except Exception as e:
                print(f"Warning: Could not save frame {frame_path_saved}. Error: {e}")


    print(f"Extracted and saved a total of {len(all_frame_info_list)} frames.")

    if not all_frame_info_list:
        print("No frames extracted from any video. Cannot proceed with training or evaluation. Exiting.")
        return # Script terminates if no frames are extracted

    # --- 图片输出 2b: Frame Distribution ---
    print("Saving frame distribution plot...")
    df_frame_info = pd.DataFrame(all_frame_info_list)
    if 'label_str' in df_frame_info.columns and not df_frame_info.empty:
        frame_counts = df_frame_info.groupby(['label_str', 'mode']).size().unstack(fill_value=0)
        frame_counts = frame_counts.reindex(CLASSES, fill_value=0)
        valid_modes = [mode for mode in ['WL', 'NBI'] if mode in frame_counts.columns]
        frame_counts = frame_counts[valid_modes] if valid_modes else pd.DataFrame(index=CLASSES)

        if not frame_counts.empty and not frame_counts.sum().sum() == 0:
            ax = frame_counts.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis')
            plt.title('Number of Frames per Class and Mode (after extraction)')
            plt.xlabel('Class')
            plt.ylabel('Number of Frames')
            plt.xticks(rotation=0)
            plt.legend(title='Mode')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            frame_distribution_plot_path = os.path.join(OUTPUT_DIR, 'frame_distribution.png')
            plt.savefig(frame_distribution_plot_path)
            print(f"Frame distribution plot saved to {frame_distribution_plot_path}")
            plt.close()
        else:
            print("No frames with valid labels extracted to plot distribution.")
    else:
         print("Could not generate frame distribution plot: No frame information collected.")


    # 4. Split dataset (stratified sampling at frame level)
    print("\nSplitting dataset...")
    # Use all_frame_info_list directly for splitting based on indices
    indices = list(range(len(all_frame_info_list)))
    labels_for_split = [info['label'] for info in all_frame_info_list]

    # Check if there are enough samples for stratification (at least 1 per class in test/val set)
    unique_labels, label_counts = np.unique(labels_for_split, return_counts=True)
    min_samples_total = len(all_frame_info_list)
    # Calculate required minimum samples for stratified split in test/val set
    min_test_size_needed = len(CLASSES) if len(unique_labels) == len(CLASSES) else 1 # Need at least 1 per class if all classes present, else at least 1
    min_val_size_needed = len(CLASSES) if len(unique_labels) == len(CLASSES) else 1


    # Determine test set size ensuring enough samples for stratification
    # test_size should be at least min_test_size_needed
    test_size = max(min_test_size_needed, int(min_samples_total * (1 - TRAIN_VALID_SPLIT_RATIO)))
    # Adjust test_size if it's too large relative to total samples
    if test_size >= min_samples_total:
         # If test_size is >= total, make train_val have at least min_val_size_needed or 1, and test gets the rest
         remaining_for_train_val = min_samples_total - max(min_val_size_needed, 1)
         if remaining_for_train_val <= 0: # Cannot split if total is too small
              print(f"Error: Total samples ({min_samples_total}) too small for split with {len(CLASSES)} classes and min requirements.")
              return # Script terminates
         test_size = min_samples_total - remaining_for_train_val

    # Perform the first split: Train+Validation vs Test
    # Use stratify=labels_for_split only if all target classes are present in labels_for_split
    stratify_labels = labels_for_split if len(unique_labels) == len(CLASSES) else None

    if stratify_labels is not None:
        train_val_indices, test_indices, y_train_val, y_test_dummy = train_test_split(
            indices, labels_for_split,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=RANDOM_SEED
        )
    else:
         print("Warning: Not all target classes present in dataset. Skipping stratification for first split.")
         train_val_indices, test_indices, y_train_val, y_test_dummy = train_test_split(
            indices, labels_for_split,
            test_size=test_size,
            # stratify=None, # No stratification
            random_state=RANDOM_SEED
        )


     # Second split: Train vs Validation
    train_val_samples_count = len(train_val_indices)
    # Determine validation set size ensuring enough samples for stratification
    # val_size should be at least min_val_size_needed
    val_size = max(min_val_size_needed, int(train_val_samples_count * VALID_TEST_SPLIT_RATIO))
    # Adjust val_size if it's too large relative to train_val samples
    if val_size >= train_val_samples_count:
        # If val_size is >= train_val count, train must have at least 1 sample
        val_size = max(min_val_size_needed, train_val_samples_count - 1)
        if val_size < min_val_size_needed:
            print(f"Error: Train+Validation samples ({train_val_samples_count}) too small for split with {len(CLASSES)} classes and min requirements.")
            return # Script terminates


    # Use stratify=y_train_val only if all target classes are present in y_train_val
    unique_labels_train_val = np.unique(y_train_val)
    stratify_y_train_val = y_train_val if len(unique_labels_train_val) == len(CLASSES) else None

    if stratify_y_train_val is not None:
        train_indices, val_indices, y_train_dummy, y_val_dummy = train_test_split(
            train_val_indices, y_train_val, # Stratify based on the labels of the train_val subset
            test_size=val_size,
            stratify=stratify_y_train_val,
            random_state=RANDOM_SEED
        )
    else:
         print("Warning: Not all target classes present in train+validation set. Skipping stratification for second split.")
         train_indices, val_indices, y_train_dummy, y_val_dummy = train_test_split(
            train_val_indices, y_train_val,
            test_size=val_size,
            # stratify=None, # No stratification
            random_state=RANDOM_SEED
        )


    # Reconstruct frame info lists based on indices
    train_frame_info = [all_frame_info_list[i] for i in train_indices]
    val_frame_info = [all_frame_info_list[i] for i in val_indices]
    test_frame_info = [all_frame_info_list[i] for i in test_indices]

    print(f"Dataset split: Train ({len(train_frame_info)} frames), Validation ({len(val_frame_info)} frames), Test ({len(test_frame_info)} frames)")

    # Exit if any split is empty (prevents errors in DataLoader)
    if not train_frame_info or not val_frame_info or not test_frame_info:
        print("Error: Dataset split resulted in an empty set. Please check split ratios and data size. Exiting.")
        return


    # 5. Define image transforms (Preprocessing and Data Augmentation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomRotation(15), # Random rotation up to 15 degrees
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), # Random crop and resize
            transforms.RandomHorizontalFlip(), # Random horizontal flip
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Color jitter
            transforms.ToTensor(), # HWC (uint8) -> CHW (float32), pixel values [0, 1]
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet mean and std for transfer learning
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
         'test': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    # --- 图片输出 3: Data Augmentation Examples ---
    print("\nSaving data augmentation examples...")
    if train_frame_info: # Ensure training set is not empty
        # Randomly select one training image's info
        example_img_info = random.choice(train_frame_info)
        example_img_path = example_img_info['img_path']
        try:
            original_img = Image.open(example_img_path).convert('RGB')

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 3, 1)
            plt.imshow(original_img)
            plt.title('Original')
            plt.axis('off')

            # Apply data augmentation several times and display
            for i in range(5):
                # Create a fresh copy of the original image for each augmentation
                augmented_img_tensor = data_transforms['train'](original_img.copy())
                # ToTensor results in [0, 1], Normalize results in [-2, 2] approx
                # For displaying: undo Normalize, then CHW to HWC
                augmented_img_display = augmented_img_tensor.clone()
                # Undo normalization: pixel_norm = (pixel_orig - mean) / std  => pixel_orig = pixel_norm * std + mean
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                augmented_img_display = augmented_img_display * std + mean
                # Clamp values to [0, 1] and convert CHW to HWC for display
                augmented_img_display = np.clip(augmented_img_display.permute(1, 2, 0).numpy(), 0, 1)

                plt.subplot(2, 3, i + 2)
                plt.imshow(augmented_img_display)
                plt.title(f'Augmented {i+1}')
                plt.axis('off')

            augmentation_plot_path = os.path.join(OUTPUT_DIR, 'data_augmentation_examples.png')
            plt.tight_layout()
            plt.savefig(augmentation_plot_path)
            print(f"Data augmentation examples saved to {augmentation_plot_path}")
            plt.close()
        except Exception as e:
             print(f"Warning: Could not generate data augmentation examples from {example_img_path}. Error: {e}")
    else:
        print("No training data available to show augmentation examples.")


    # 6. Create DataLoaders
    train_dataset = ColonoscopyFrameDataset(train_frame_info, transform=data_transforms['train'])
    val_dataset = ColonoscopyFrameDataset(val_frame_info, transform=data_transforms['val'])
    test_dataset = ColonoscopyFrameDataset(test_frame_info, transform=data_transforms['test'])

    # Set shuffle=False for val and test loaders to ensure consistent order for evaluation
    # *** Use custom_collate_fn for all DataLoaders ***
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)

    # 7. Build Model
    print("\nBuilding model...")
    model = build_model(MODEL_NAME, len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Or optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 8. Train Model
    print("\nStarting training...")
    # Pass MODEL_SAVE_PATH to the train_model function
    model_trained, train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE, model_save_path=MODEL_SAVE_PATH
    )

    # --- 图片输出 4: Training Plot ---
    print("Saving training plot...")
    if NUM_EPOCHS > 0 and (train_losses or val_losses): # Only plot if training ran for at least one epoch
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(NUM_EPOCHS), train_losses, label='Train Loss')
        plt.plot(range(NUM_EPOCHS), val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(NUM_EPOCHS), train_accs, label='Train Accuracy')
        plt.plot(range(NUM_EPOCHS), val_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        training_plot_path = os.path.join(OUTPUT_DIR, 'training_plot.png')
        plt.tight_layout()
        plt.savefig(training_plot_path)
        print(f"Training plot saved to {training_plot_path}")
        plt.close()
    else:
        print("Skipping training plot: No epochs were trained.")


    # 9. Evaluate Model and Output Images
    print("\nEvaluating model on test set...")
    # evaluate_model now returns data only for successfully processed samples
    cm, report_dict, all_labels, all_preds, all_probs, test_image_info_evaluated = evaluate_model(model_trained, test_loader, device=DEVICE)
    # confusion matrix, ROC curve, and Precision-Recall curve are saved within evaluate_model function

    # 10. Grad-CAM Visualization and Output Images
    print("\nGenerating Grad-CAM visualizations...")
    # --- IMPORTANT: Set this to the name of the LAST CONVOLUTIONAL LAYER in your chosen model ---
    # You need to inspect your model architecture to find the correct name.
    # For ResNet models, common candidates are the last block or the last conv layer within it.
    # Example names (may vary slightly based on torchvision version):
    # ResNet18/34: 'layer4' or 'layer4.1.conv2' (if using BasicBlock)
    # ResNet50/101/152: 'layer4' or 'layer4.2.conv3' (if using Bottleneck)
    # Print your model structure (print(model_trained)) and find the appropriate layer name.
    # print(model_trained) # Uncomment this line to print model structure
    gradcam_target_layer = 'layer4' # Default guess for ResNet, CHANGE THIS IF NECESSARY based on print(model_trained)!
    # ------------------------------------------------------------------------------------------

    # Proceed with Grad-CAM only if evaluation was successful (samples processed)
    # and the number of samples matches the collected info items.
    if gradcam_target_layer and len(all_labels) > 0 and len(all_labels) == len(test_image_info_evaluated):
        try:
            print(f"Attempting Grad-CAM with target layer: '{gradcam_target_layer}'")
            # Initialize GradCAM object (hooks are registered later just before use)
            grad_cam = GradCAM(model_trained, gradcam_target_layer)

            # Select samples from the test set for visualization
            num_gradcam_examples_per_status = 3 # Save this many correct/incorrect examples per class

            # Collect indices of successfully evaluated test samples by their true label and prediction status
            test_sample_indices_by_class_status = {cls: {'correct': [], 'incorrect': []} for cls in CLASSES}

            # Iterate through the collected info list (which now matches the prediction lists)
            # and categorize sample indices.
            for i in range(len(test_image_info_evaluated)):
                img_info = test_image_info_evaluated[i]
                true_label_idx = img_info['label'] # Use label from the info dict
                # Ensure the true label index is valid and map to string
                if true_label_idx not in IDX_TO_CLASS:
                     print(f"Warning: Skipping sample index {i} for Grad-CAM categorization due to invalid true label index: {true_label_idx}")
                     continue
                true_label_str = IDX_TO_CLASS[true_label_idx]

                predicted_label_idx = all_preds[i] # Get prediction from the all_preds list which aligns with test_image_info_evaluated

                status = 'correct' if true_label_idx == predicted_label_idx else 'incorrect'

                # Ensure the true class is in our target CLASSES
                if true_label_str in CLASSES:
                    test_sample_indices_by_class_status[true_label_str][status].append(i) # Append index 'i' from test_image_info_evaluated


            gradcam_count_saved = {cls: {'correct': 0, 'incorrect': 0} for cls in CLASSES}

            print("Selecting and generating samples for Grad-CAM...")
            # Iterate through each class and each status (correct/incorrect)
            for cls in CLASSES:
                for status in ['correct', 'incorrect']:
                    indices_pool = test_sample_indices_by_class_status[cls][status]
                    num_samples_to_save = min(len(indices_pool), num_gradcam_examples_per_status)
                    if num_samples_to_save == 0:
                         # print(f"No {status} samples found for class {cls} in successfully evaluated set.") # Optional
                         continue

                    # Randomly sample indices from the pool
                    selected_indices = random.sample(indices_pool, num_samples_to_save)

                    print(f"Generating {num_samples_to_save} Grad-CAM examples for Class: {cls}, Status: {status}...")

                    for i in selected_indices: # 'i' here is the index within test_image_info_evaluated (and all_preds, all_labels)
                        img_info = test_image_info_evaluated[i] # Retrieve the info dictionary

                        # Double check img_info validity before proceeding (should be ok now with collected list)
                        if not (isinstance(img_info, dict) and 'img_path' in img_info and 'label_str' in img_info and 'case_id' in img_info and 'mode' in img_info and 'frame_idx' in img_info):
                             print(f"Warning: Skipping Grad-CAM generation for item at index {i} due to invalid info format: {img_info}")
                             continue


                        true_label_str = img_info['label_str']
                        predicted_label_idx = all_preds[i] # Use the prediction corresponding to this index 'i'
                        predicted_label_str = IDX_TO_CLASS[predicted_label_idx]

                        # Path to the saved extracted frame image
                        img_path_saved = img_info['img_path']

                        # Load the image and apply the test transform to get the tensor input for the model
                        # This step is repeated because Grad-CAM needs the raw image data and a tensor with requires_grad=True
                        # It's safer to reload and re-transform for Grad-CAM specifically.
                        try:
                            img_pil = Image.open(img_path_saved).convert('RGB')
                            if img_pil is None:
                                print(f"Warning: Could not load image {img_path_saved} for Grad-CAM transformation.")
                                continue
                            # Apply the test transform
                            input_tensor = data_transforms['test'](img_pil)
                            # Add batch dimension and move to device
                            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
                             # print(f"Successfully loaded and transformed {img_path_saved} for Grad-CAM.") # Debugging
                        except Exception as e:
                             print(f"Warning: Error loading or transforming image {img_path_saved} for Grad-CAM. {e}")
                             continue # Skip this sample for Grad-CAM

                        # Generate Grad-CAM heatmap for the predicted class of this sample
                        try:
                           # Pass the predicted class as the target class for visualization
                           heatmap = grad_cam.generate_heatmap(input_tensor, target_class=predicted_label_idx)
                           if heatmap is None:
                               print(f"Warning: Heatmap generation failed for {img_path_saved}. Skipping.")
                               continue # Skip this sample if heatmap failed
                           # print(f"Successfully generated heatmap for {img_path_saved}.") # Debugging
                        except Exception as e:
                           print(f"Warning: Error generating heatmap for {img_path_saved}. {e}")
                           import traceback
                           traceback.print_exc()
                           continue # Skip this sample if heatmap generation failed

                        # Save the superimposed image
                        # Filename: TrueClass_PredClass_CaseID_Mode_frameXXX_gradcam.jpg
                        img_save_name = f"True{true_label_str}_Pred{predicted_label_str}_{img_info['case_id']}_{img_info['mode']}_frame{img_info['frame_idx']:03d}_gradcam.jpg"
                        save_path = os.path.join(OUTPUT_DIR, 'grad_cam', img_save_name)

                        # save_gradcam_image needs the path to the saved extracted frame, heatmap,
                        # predicted and true labels strings, and the img_info dictionary.
                        try:
                            success = grad_cam.save_gradcam_image(img_path_saved, heatmap, predicted_label_str, true_label_str, img_info, save_path)
                            if success:
                                # print(f"Saved Grad-CAM for {img_save_name}") # Optional: too verbose
                                gradcam_count_saved[true_label_str][status] += 1
                        except Exception as e:
                             print(f"Warning: Error saving Grad-CAM image {save_path}. {e}")


            print(f"Finished generating Grad-CAM for all selected samples.")


        except RuntimeError as e:
             print(f"\nGrad-CAM Runtime Error: {e}")
             print(f"Please check if the 'gradcam_target_layer = '{gradcam_target_layer}' ' setting in the script is correct for your chosen model ('{MODEL_NAME}').")
             # print(model_trained) # Uncomment this line to print model structure
        except Exception as e:
            print(f"\nAn unexpected error occurred during Grad-CAM visualization: {e}")
            import traceback
            traceback.print_exc()
        finally:
             # Ensure hooks are removed after the entire Grad-CAM process is done
             if 'grad_cam' in locals() and grad_cam is not None:
                 try:
                      grad_cam.remove_hooks()
                      # print("Final Grad-CAM hooks cleanup done.") # Optional
                 except Exception as e:
                      print(f"Warning: Error during final Grad-CAM hooks cleanup. {e}")

    else:
        if not gradcam_target_layer:
            print("\nSkipping Grad-CAM: gradcam_target_layer is not set.")
        elif len(all_labels) == 0:
            print("\nSkipping Grad-CAM: No samples were successfully evaluated.")
        elif len(all_labels) != len(test_image_info_evaluated):
             print(f"\nSkipping Grad-CAM: Mismatch between number of evaluated samples ({len(all_labels)}) and collected info items ({len(test_image_info_evaluated)}).")
             print("Please fix the data loading/info collection issue.")


    # 11. Clean up extracted frames (Optional)
    # print(f"\nCleaning up extracted frames directory: {frames_save_dir}")
    # shutil.rmtree(frames_save_dir) # Uncomment this if you want to remove the extracted frames after the run


if __name__ == "__main__":
    # The KMP_DUPLICATE_LIB_OK environment variable is set at the top of the script
    print(f"Running script on device: {DEVICE}")
    main()
    print("\nScript finished.")