from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Tuple, Optional, List
import numpy as np


@dataclass
class GroupRegion:
    centroid: Tuple[float, float] = (0.0, 0.0)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    word_positions: List[Tuple[int, int]] = field(default_factory=list)
    total_area: int = 0
    
    def update(self, position: Tuple[int, int], box_size: Tuple[int, int]):
        x, y = position
        w, h = box_size
        area = w * h
        
        old_count = len(self.word_positions)
        if old_count == 0:
            self.centroid = (x + w / 2, y + h / 2)
        else:
            cx, cy = self.centroid
            new_cx = (cx * self.total_area + (x + w / 2) * area) / (self.total_area + area)
            new_cy = (cy * self.total_area + (y + h / 2) * area) / (self.total_area + area)
            self.centroid = (new_cx, new_cy)
        
        self.total_area += area
        self.word_positions.append(position)
        
        x_min, y_min, x_max, y_max = self.bounding_box
        self.bounding_box = (
            min(x_min, x) if x_min > 0 else x,
            min(y_min, y) if y_min > 0 else y,
            max(x_max, x + w),
            max(y_max, y + h)
        )


class GroupAwarePlacer:
    def __init__(self, mode: str, radius: int, strength: float):
        self.mode = mode
        self.radius = radius
        self.strength = strength
        self.group_regions: Dict[Hashable, GroupRegion] = {}
    
    def get_placement_position(
        self,
        group_id: Optional[Hashable],
        box_size: Tuple[int, int],
        occupancy_map,
        random_state
    ) -> Optional[Tuple[int, int]]:
        if group_id is None or group_id not in self.group_regions:
            return occupancy_map.sample_position(box_size[0], box_size[1], random_state)
        
        if self.mode == 'strict':
            return self._strict_placement(group_id, box_size, occupancy_map, random_state)
        else:
            return self._probabilistic_placement(group_id, box_size, occupancy_map, random_state)
    
    def _strict_placement(self, group_id, box_size, occupancy_map, random_state):
        region = self.group_regions[group_id]
        cx, cy = region.centroid
        
        x_min = max(0, int(cx - self.radius))
        x_max = min(occupancy_map.height, int(cx + self.radius) + box_size[0])
        y_min = max(0, int(cy - self.radius))
        y_max = min(occupancy_map.width, int(cy + self.radius) + box_size[1])
        
        if x_min >= x_max or y_min >= y_max:
            return occupancy_map.sample_position(box_size[0], box_size[1], random_state)
        
        result = self._query_region(
            occupancy_map.integral,
            box_size[0], box_size[1],
            x_min, x_max, y_min, y_max,
            random_state
        )
        
        if result is None:
            return occupancy_map.sample_position(box_size[0], box_size[1], random_state)
        
        return result
    
    def _probabilistic_placement(self, group_id, box_size, occupancy_map, random_state):
        region = self.group_regions[group_id]
        cx, cy = region.centroid
        
        positions = self._find_all_valid_positions(
            occupancy_map.integral, box_size[0], box_size[1]
        )
        
        if len(positions) == 0:
            return None
        
        distances = np.sqrt(
            (positions[:, 0] - cx) ** 2 +
            (positions[:, 1] - cy) ** 2
        )
        
        weights = 1.0 / (1.0 + distances)
        uniform = np.ones(len(weights)) / len(weights)
        final_weights = (1 - self.strength) * uniform + self.strength * weights
        final_weights /= final_weights.sum()
        
        idx = random_state.choices(range(len(positions)), weights=final_weights, k=1)[0]
        return tuple(positions[idx])
    
    def _query_region(self, integral, size_x, size_y, x_min, x_max, y_min, y_max, random_state):
        height, width = integral.shape
        valid_positions = []
        
        for i in range(x_min, min(x_max, height - size_x)):
            for j in range(y_min, min(y_max, width - size_y)):
                area = integral[i, j] + integral[i + size_x, j + size_y]
                area -= integral[i + size_x, j] + integral[i, j + size_y]
                if not area:
                    valid_positions.append((i, j))
        
        if not valid_positions:
            return None
        
        return valid_positions[random_state.randint(0, len(valid_positions))]
    
    def _find_all_valid_positions(self, integral, size_x, size_y):
        height, width = integral.shape
        positions = []
        
        for i in range(height - size_x):
            for j in range(width - size_y):
                area = integral[i, j] + integral[i + size_x, j + size_y]
                area -= integral[i + size_x, j] + integral[i, j + size_y]
                if not area:
                    positions.append((i, j))
        
        return np.array(positions) if positions else np.array([]).reshape(0, 2)
    
    def update_group(self, group_id: Hashable, position: Tuple[int, int], 
                     box_size: Tuple[int, int]):
        if group_id not in self.group_regions:
            self.group_regions[group_id] = GroupRegion()
        
        self.group_regions[group_id].update(position, box_size)
