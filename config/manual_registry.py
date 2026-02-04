"""
Author: Tamasa Patra
Purpose: Centralized manual registry for multi-equipment system
"""

import os
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ManualDefinition:
    """Definition for a single equipment manual."""
    pdf_path: str
    equipment_type: str
    equipment_brand: str
    equipment_model: str
    manual_type: str
    title: str
    language: str = "en"
    tier: int = 1  # Priority tier (1=highest)
    
    def to_metadata(self) -> Dict:
        """Convert to ChromaDB metadata format."""
        return {
            "source": os.path.basename(self.pdf_path),
            "title": self.title,
            "equipment_type": self.equipment_type,
            "equipment_brand": self.equipment_brand,
            "equipment_model": self.equipment_model,
            "manual_type": self.manual_type,
            "language": self.language,
            "tier": self.tier
        }
    
    def exists(self) -> bool:
        """Check if PDF file exists."""
        return os.path.exists(self.pdf_path)


class ManualRegistry:
    """Registry of all available equipment manuals."""
    
    def __init__(self):
        """Initialize manual registry."""
        self.manuals = self._load_manuals()
    
    def _load_manuals(self) -> List[ManualDefinition]:
        """
        Define all available manuals.
        """
        manuals = [
            # ========================================
            # TIER 1: POS SYSTEMS
            # ========================================
            ManualDefinition(
                pdf_path="Manuals/pos/V400m POS Manual.pdf",
                equipment_type="POS",
                equipment_brand="V400m",
                equipment_model="V400m",
                manual_type="software",
                title="V400m POS Manual",
                tier=3
            ),
            
            # ========================================
            # TIER 1: COFFEE EQUIPMENT
            # ========================================
            ManualDefinition(
                pdf_path="Manuals/coffee/Coffee Maker Manual.pdf",
                equipment_type="Coffee_Maker",
                equipment_brand="Metos",
                equipment_model="M200 / MT200",
                manual_type="operation",
                title="Coffee Maker Manual",
                tier=1
            ),
            
            # ========================================
            # TIER 2: KITCHEN EQUIPMENT
            # ========================================
            ManualDefinition(
                pdf_path="Manuals/kitchen/Vulcan Installation & Operation Manual.pdf",
                equipment_type="Oven",
                equipment_brand="Vulcan",
                equipment_model="VC4GD",
                manual_type="operation",
                title="Vulcan Installation & Operation Manual",
                tier=2
            ),
            ManualDefinition(
                pdf_path="Manuals/kitchen/Pitco Fryer Manual.pdf",
                equipment_type="Fryer",
                equipment_brand="Pitco",
                equipment_model="SG14",
                manual_type="operation",
                title="Pitco Fryer Manual",
                tier=2
            )
        ]
        return manuals
    
    def get_all_manuals(self) -> List[ManualDefinition]:
        return self.manuals
    
    def get_available_manuals(self) -> List[ManualDefinition]:
        return [m for m in self.manuals if m.exists()]
    
    def get_missing_manuals(self) -> List[ManualDefinition]:
        return [m for m in self.manuals if not m.exists()]
    
    def get_by_equipment_type(self, equipment_type: str) -> List[ManualDefinition]:
        return [m for m in self.manuals if m.equipment_type == equipment_type]
    
    def get_by_tier(self, tier: int) -> List[ManualDefinition]:
        return [m for m in self.manuals if m.tier == tier]
    
    def validate_manuals(self) -> Dict:
        available = self.get_available_manuals()
        missing = self.get_missing_manuals()
        return {
            "total_registered": len(self.manuals),
            "available": len(available),
            "missing": len(missing),
            "available_manuals": [{"title": m.title, "path": m.pdf_path, "equipment_type": m.equipment_type} for m in available],
            "missing_manuals": [{"title": m.title, "path": m.pdf_path, "equipment_type": m.equipment_type} for m in missing]
        }


# Singleton instance
manual_registry = ManualRegistry()
