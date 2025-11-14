# Further finetune previous finetuned Resnet model on Cifar-10
import os, re, torch, pandas as pd
from datetime import datetime
# Optional progress bars for better within-epoch visibility
try:
    from tqdm.auto import tqdm
    _TQDM = True
except Exception:
    _TQDM = False
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.datasets import CIFAR10
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

"""
Project training hyperparameters

This block defines dataset paths, image size, batch size, number of epochs, learning rate and device selection. 
These parameters were chosen to suit a typical laptop GPU environment used in the mini-project:

IMG_SIZE = 224: 
    matches the input size expected by ResNet pretrained on ImageNet 
    (keeps compatibility with pretrained weights and common image augmentations).

BATCH = 32: 
    a compromise between throughput and memory use. 
    For faster iteration on CPU-only machines lower values are necessary.
 
EPOCHS = 15: 
    a modest number for fine-tuning; 
    long enough to converge for the assignment but short enough to run on limited hardware.

LR = 3e-4: 
    a conservative learning rate for AdamW when fine-tuning a pretrained network. 
    It balances stable training with reasonable progress.

Tweak notes: if you have a high-memory GPU you can increase BATCH and/or
IMG_SIZE for potentially better accuracy; on low-memory devices reduce BATCH
and consider using gradient accumulation.
"""

DATA_ROOT = "datasets"
IMG_SIZE = 224
BATCH = 32
EPOCHS = 15
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TRAIN_SAMPLES = 4000  # dataset cap: None to disable
MAX_VAL_SAMPLES = 1000    # dataset cap: None to disable

mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
"""
Data transforms / augmentations
- Normalization uses ImageNet mean/std because we're fine-tuning a model
  pretrained on ImageNet; this keeps input statistics similar to pretraining.
- For training we use random resized crops, flips and small rotations to
  increase robustness to scale/orientation variations in a small dataset.
- For evaluation we use a deterministic resize+center crop to produce
  consistent inputs for validation and test-time inference.
"""
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def main():
    # Set up a timestamped training report file and a helper that prints and writes
    os.makedirs("reports", exist_ok=True)
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join("reports", f"training-cifar_{_ts}.txt")

    def log(msg: str = ""):
        print(msg)
        try:
            with open(report_path, "a", encoding="utf-8") as rf:
                rf.write(str(msg) + "\n")
        except Exception:
            pass

    log(f"Training report started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Show device information
    log(f"Using device: {DEVICE}")
    if DEVICE.startswith("cuda"):
        try:
            dev = torch.cuda.current_device()
        except Exception:
            dev = 0
        name = torch.cuda.get_device_name(dev)
        cap = torch.cuda.get_device_capability(dev)
        total_mem_gb = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        log(f"GPU: {name} | Compute Capability: {cap[0]}.{cap[1]} | VRAM: {total_mem_gb:.1f} GB")

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = True if DEVICE.startswith("cuda") else False
    # Log settings
    log("Parameters:")
    log(f"  DATA_ROOT={DATA_ROOT}")
    log(f"  IMG_SIZE={IMG_SIZE}  BATCH={BATCH}  EPOCHS={EPOCHS}  LR={LR}")
    log(f"  num_workers={num_workers}  pin_memory={pin_memory}")

    # Load and preprocess cifar-10 dataset
    # Retrieve cifar-10 from datasets/cifar-10-batches-py
    trainval_full = CIFAR10(root=DATA_ROOT, train=True, transform=None, download=False)  # change download to True to download from cloud
    class_names = trainval_full.classes  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    g = torch.Generator().manual_seed(42)  # ensures same split every run
    train_ds_full, val_ds_full = random_split(trainval_full, [45000, 5000], generator=g)  # create validation set from subset of train set

    # ---- MAKE TRAINING SET IMBALANCED ----
    # Minority classes: bird (2), deer (4), frog (6)
    minority_classes = [2, 4, 6]
    reduction_factor = 0.2  # keep only 20% of samples for minority classes

    original_train_indices = train_ds_full.indices  # indices into trainval_full
    targets = trainval_full.targets

    new_train_indices = []
    per_class_before = [0] * 10
    per_class_after = [0] * 10

    # Count original per-class sizes within the training split
    for idx in original_train_indices:
        lbl = targets[idx]
        per_class_before[lbl] += 1

    # Build new (imbalanced) index list
    for c in range(10):
        # all indices in the training split that belong to class c
        class_idx_list = [idx for idx in original_train_indices if targets[idx] == c]

        if c in minority_classes:
            keep_n = max(1, int(len(class_idx_list) * reduction_factor))
            # deterministic: take first keep_n to keep it simple and reproducible
            kept = class_idx_list[:keep_n]
        else:
            kept = class_idx_list

        new_train_indices.extend(kept)
        per_class_after[c] = len(kept)

    # Replace training subset with the imbalanced subset
    from torch.utils.data import Subset
    train_ds_full = Subset(trainval_full, new_train_indices)

    # Log distribution before/after
    log("Train class counts BEFORE imbalance:")
    for c in range(10):
        log(f"  class {c} ({class_names[c]}): {per_class_before[c]}")

    log("Train class counts AFTER imbalance:")
    for c in range(10):
        log(f"  class {c} ({class_names[c]}): {per_class_after[c]}")

    # Attach transforms
    train_ds_full.dataset.transform = train_tfms
    val_ds_full.dataset.transform = eval_tfms
    
    # Limit dataset size, same as in initial finetuning
    torch.manual_seed(42)
    train_ds = (torch.utils.data.Subset(train_ds_full, torch.randperm(len(train_ds_full))[:MAX_TRAIN_SAMPLES].tolist())
                if MAX_TRAIN_SAMPLES and len(train_ds_full) > MAX_TRAIN_SAMPLES else train_ds_full)
    val_ds = (torch.utils.data.Subset(val_ds_full, torch.randperm(len(val_ds_full))[:MAX_VAL_SAMPLES].tolist())
              if MAX_VAL_SAMPLES and len(val_ds_full) > MAX_VAL_SAMPLES else val_ds_full)
    
     # ---- Compute class weights from the (imbalanced) training set ----
    num_classes = 10
    label_counts = torch.zeros(num_classes, dtype=torch.long)

    count_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    for _, labels in count_loader:
        for l in labels:
            label_counts[l.item()] += 1

    total_samples = int(label_counts.sum().item())
    class_weights = (total_samples / (num_classes * label_counts.float())).to(DEVICE)

    log(f"Train class counts used for weights: {label_counts.tolist()}")
    log(f"Train class weights: {class_weights.tolist()}")

    # Part 4: Model training and learning optimization part
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Use persistent workers and small prefetch to speed up data pipeline when workers > 0
    loader_kwargs = dict(batch_size=BATCH, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    train_dl = DataLoader(train_ds, shuffle=True, **loader_kwargs) # training needs randomization
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Load previously finetuned model and saved weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 10) #set to 10 as Cifar-10 has 10 classes

    # Load previously fine-tuned weights 
    state_dict = torch.load("best.pt", map_location=DEVICE, weights_only=True)
    # Filter out the classifier layer weights
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    model.load_state_dict(filtered_state_dict, strict=False)

    model = model.to(DEVICE)
    
    # retrieve test dataset from datasets/cifar-10-batches-py
    test_ds = CIFAR10(root=DATA_ROOT, train=False, transform=eval_tfms, download=False)
    # Build test dataset and DataLoader
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
     # Conduct inference on cifar-10 test set
    model.eval()
    all_preds = []
    all_labels = []
    rows = []

    with torch.inference_mode():
        for batch_idx, (xb, labels) in enumerate(test_dl):
            xb = xb.to(DEVICE)
            labels = labels.to(DEVICE)
        
            logits = model(xb)
            preds = logits.argmax(1)
        
            # Collect metrics
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
            # Compute submission IDs
            start_idx = batch_idx * BATCH
            ids = list(range(start_idx, start_idx + len(preds)))
        
            # Build rows for CSV
            rows += [{"id": i, "predicted": p.item()} for i, p in zip(ids, preds)]
    
    # showcase testing results with evaluation metrics
    log("Testing results BEFORE additional training")
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    log(f"accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    log("confusion_matrix:")
    log(cm)
    
    pd.DataFrame(rows).to_csv("submission-cifar-before-training.csv", index=False)
    log("Wrote submission-cifar-before-training.csv")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Mixed precision setup: enables faster,smaller training on GPU
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_amp else None

    best_acc, best_path = 0.0, "best-cifar.pt"

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        batch_count = 0
        train_iter = tqdm(train_dl, desc=f"Train {epoch+1}/{EPOCHS}", leave=False) if _TQDM else train_dl
        for xb, yb in train_iter:
            xb = xb.to(DEVICE, non_blocking=pin_memory)
            yb = yb.to(DEVICE, non_blocking=pin_memory)
            opt.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            running_loss += loss.item()
            batch_count += 1

            # Update progress bar
            if _TQDM:
                lr_now = opt.param_groups[0]['lr']
                train_iter.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_now:.2e}"})

        avg_train_loss = running_loss / max(1, batch_count)

        # Validation
        model.eval()
        correct = total = 0
        val_loss = 0.0
        val_batches = 0
        all_preds, all_labels = [], []
        with torch.inference_mode():
            val_iter = tqdm(val_dl, desc=f"Val {epoch+1}/{EPOCHS}", leave=False) if _TQDM else val_dl
            for xb, yb in val_iter:
                xb = xb.to(DEVICE, non_blocking=pin_memory)
                yb = yb.to(DEVICE, non_blocking=pin_memory)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)

                val_loss += loss.item()
                val_batches += 1

                preds = logits.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(yb.cpu().tolist())

                # Update validation progress with batch loss and running accuracy
                if _TQDM:
                    running_acc = (correct / total) if total else 0.0
                    val_iter.set_postfix({"val_loss": f"{loss.item():.4f}", "acc": f"{running_acc:.3f}"})

        avg_val_loss = val_loss / max(1, val_batches)
        acc = correct / total if total else 0.0

        # scheduler step
        sched.step()

        # other testing metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0) # for multiclass
        cm = confusion_matrix(all_labels, all_preds)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

        log("Additional training on Cifar-10")
        log(f"epoch {epoch+1}/{EPOCHS}")
        log(f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_acc={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
        log("confusion_matrix:")
        log(cm)
            
    log(f"Best val acc: {best_acc:.3f}")
            
    # Test newly trained model again on cifar-10 test set       
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []
    rows = []

    with torch.inference_mode():
        for batch_idx, (xb, labels) in enumerate(test_dl):
            xb = xb.to(DEVICE)
            labels = labels.to(DEVICE)
        
            logits = model(xb)
            preds = logits.argmax(1)
        
            # Collect metrics
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
            # Compute submission IDs
            start_idx = batch_idx * BATCH
            ids = list(range(start_idx, start_idx + len(preds)))
        
            # Build rows for CSV
            rows += [{"id": i, "predicted": p.item()} for i, p in zip(ids, preds)]
            
    # showcase testing results with evaluation metrics
    log("Testing results AFTER additional training")
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    log(f"accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    log("confusion_matrix:")
    log(cm)

    pd.DataFrame(rows).to_csv("submission-cifar-after-training.csv", index=False)
    log("Wrote submission-cifar-after-training.csv")
    log(f"Training report saved to {report_path}")

if __name__ == "__main__":
    main()
