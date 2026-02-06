import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Add parent directory to import CubiCasa5k
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CubiCasa5k.floortrans.loaders.house import House

CUSTOM_ROOM_MAP = {
    "Closet": 1, "Storage": 1, "DressingRoom": 1, "Pantry": 1,
    "Bath": 2, "Sauna": 2, "Toilet": 2,
    "Kitchen": 3, "LivingRoom": 3, "Dining": 3, "FamilyRoom": 3, "Lounge": 3, "Office": 3, "Garage": 3,
    "Bedroom": 4, "GuestRoom": 4,
    "Hall": 5, "Entry": 5, "Corridor": 5, "StairWell": 5,
    "Outdoor": 6, "Balcony": 6, "Terrace": 6,
    "Wall": 10, "Railing": 10, 
    "Background": 0, "Undefined": 6,
    # Additional room types mapped to Background (0) to suppress warnings
    "Alcove": 0, "Attic": 0, "Ballroom": 0, "Bar": 0, "Basement": 0,
    "Below150cm": 0, "CarPort": 0, "Church": 0, "ConferenceRoom": 0,
    "Conservatory": 0, "Counter": 0, "Den": 0, "DraughtLobby": 0,
    "EatingArea": 0, "Elevated": 0, "Elevator": 0, "ExerciseRoom": 0,
    "Garbage": 0, "HallWay": 0, "HotTub": 0, "Library": 0, "Loft": 0,
    "MediaRoom": 0, "MeetingRoom": 0, "Museum": 0, "Nook": 0,
    "OpenToBelow": 0, "Reception": 0, "RecreationRoom": 0, "RetailSpace": 0,
    "Room": 0, "Sanctuary": 0, "ServiceRoom": 0, "ServingArea": 0,
    "Skylights": 0, "Stable": 0, "Stage": 0, "Stairs": 0, "SunRoom": 0,
    "SwimmingPool": 0, "TechnicalRoom": 0, "Theatre": 0, "UserDefined": 0,
    "Utility": 0,
}

CUSTOM_ICON_MAP = {
    "Window": 1,
    "Door": 2,
    "Closet": 0,
    "ClosetRound": 0,
    "ClosetTriangle": 0,
    "CoatCloset": 0,
    "CoatRack": 0,
    "CounterTop": 0,
    "Housing": 0,
    "ElectricalAppliance": 0,
    "WoodStove": 0,
    "GasStove": 0,
    "Toilet": 0,
    "Urinal": 0,
    "SideSink": 0,
    "Sink": 0,
    "RoundSink": 0,
    "CornerSink": 0,
    "DoubleSink": 0,
    "DoubleSinkRight": 0,
    "WaterTap": 0,
    "SaunaBenchHigh": 0,
    "SaunaBenchLow": 0,
    "SaunaBenchMid": 0,
    "SaunaBench": 0,
    "Fireplace": 0,
    "FireplaceCorner": 0,
    "FireplaceRound": 0,
    "PlaceForFireplace": 0,
    "PlaceForFireplaceCorner": 0,
    "PlaceForFireplaceRound": 0,
    "Bathtub": 0,
    "BathtubRound": 0,
    "Chimney": 0,
    "Misc": 0,
    "BaseCabinetRound": 0,
    "BaseCabinetTriangle": 0,
    "BaseCabinet": 0,
    "WallCabinet": 0,
    "Shower": 0,
    "ShowerCab": 0,
    "ShowerPlatform": 0,
    "ShowerScreen": 0,
    "ShowerScreenRoundRight": 0,
    "ShowerScreenRoundLeft": 0,
    "Jacuzzi": 0,
    "WashingMachine": 0,
    "IntegratedStove": 0,
    "Dishwasher": 0,
    "GeneralAppliance": 0,
    "SaunaStove": 0,
    "Cooking": 0,
    "Dryer": 0,
    "Refrigerator": 0,
    "Sofa": 0,
    "FixedFurniture": 0,
}


class DPVR_cubicasa(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        """
        root_dir: ví dụ "/kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k"
        list_file: ví dụ "train.txt"
        """
        self.root_dir = root_dir
        self.transform = transform

        self.image_file_name = "F1_scaled.png"
        self.svg_file_name = "model.svg"

        with open(os.path.join(root_dir, list_file), "r") as f:
            self.folders = [
                line.strip().strip("/")
                for line in f
                if line.strip()
            ]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        """
        Trả về dictionary gồm:
            - image: (3, H, W) float tensor
            - boundary: (H, W) long tensor (0=bg, 1=wall/railing)
            - room: (H, W) long tensor (0-6 class)
            - door: (H, W) long tensor (0=bg, 1=door/window)
        """
        sample = self.get_txt(index)
        image = sample['image']
        
        masks = [sample['boundary'], sample['room'], sample['door']]
            
        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            image = augmented['image']
            boundary = augmented['masks'][0]
            room = augmented['masks'][1]
            door = augmented['masks'][2]
        else:
            boundary, room, door = masks

        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        if not isinstance(boundary, torch.Tensor):
            boundary = torch.from_numpy(boundary).long()
            room = torch.from_numpy(room).long()
            door = torch.from_numpy(door).long()

        return {
            'image': image, 
            'boundary': boundary, 
            'room': room, 
            'door': door
        }

    def get_txt(self, index):
        """Load sample from file system."""
        folder = self.folders[index]
    
        img_path = os.path.join(self.root_dir, folder, self.image_file_name)
        svg_path = os.path.join(self.root_dir, folder, self.svg_file_name)
    
        # Read image
        fplan = cv2.imread(img_path)
        if fplan is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
    
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)
        height, width, _ = fplan.shape
    
        # Load house from SVG
        house = House(svg_path, height, width, room_list=CUSTOM_ROOM_MAP, icon_list=CUSTOM_ICON_MAP)
    
        # Build room mask from SVG
        room_mask = house.get_segmentation_tensor()[0]
        
        # Map room classes using CUSTOM_ROOM_MAP
        room_output = np.zeros((height, width), dtype=np.uint8)
        for original_class, mapped_class in CUSTOM_ROOM_MAP.items():
            room_output[room_mask == original_class] = mapped_class
    
        # Build boundary mask (walls)
        boundary_mask = np.zeros((height, width), dtype=np.uint8)
        for wall in house.wall_objs:
            boundary_mask[wall.rr, wall.cc] = 1
    
        # Build door/window mask
        door_mask = np.zeros((height, width), dtype=np.uint8)
        door_mask[(house.icons == 1) | (house.icons == 2)] = 1
        boundary_mask[door_mask == 1] = 0
    
        return {
            "image": fplan.astype(np.uint8),
            "boundary": boundary_mask,
            "room": room_output,
            "door": door_mask,
        }