
import torch 
import torchvision
import torchvision.transforms as transforms

from torchvision.transforms.functional import to_pil_image

def cifar_dataloader (data_root='./data', batch_size=128,validation_size=5000,seed=42,num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])    
    full_train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)  
    
    train_data, val_data = torch.utils.data.random_split(
        full_train_data, 
        [len(full_train_data) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True#,pin_memory=True
                                               ,num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False#,pin_memory=True
                                             )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False#,pin_memory=True
                                              )
    return train_loader, val_loader, test_loader
