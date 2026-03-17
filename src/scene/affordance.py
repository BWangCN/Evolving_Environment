"""Infer object affordances from category name.

Maps object categories to manipulation-relevant properties:
- graspable: can the robot pick this up?
- is_container: can it hold other objects inside? (bowl, cup, box)
- is_surface: can objects be placed on top? (plate, tray, table)
- valid_as_target: what task types can this object be a TARGET of?
- valid_as_object: what task types can this object be the MANIPULATED object of?
"""

from .object import SceneObject, TaskType

# Category → affordance mapping.
# Extend this dict as new object categories are introduced.
AFFORDANCE_DB: dict[str, dict] = {
    # --- Kitchen ---
    "mug": {
        "graspable": True,
        "is_container": True,
        "is_surface": False,
        "container_opening_radius": 0.04,
    },
    "bowl": {
        "graspable": True,
        "is_container": True,
        "is_surface": False,
        "container_opening_radius": 0.07,
    },
    "plate": {
        "graspable": True,
        "is_container": False,
        "is_surface": True,
    },
    "bottle": {
        "graspable": True,
        "is_container": False,  # opening too small for placing objects in
        "is_surface": False,
    },
    "spoon": {
        "graspable": True,
        "is_container": False,
        "is_surface": False,
    },
    # --- Added in E2 ---
    "toy": {
        "graspable": True,
        "is_container": False,
        "is_surface": False,
    },
    "pill_bottle": {
        "graspable": True,
        "is_container": False,
        "is_surface": False,
    },
    "dispenser": {
        "graspable": True,
        "is_container": False,
        "is_surface": False,
    },
    # --- E5 new objects ---
    "wine_glass": {
        "graspable": True,
        "is_container": True,
        "is_surface": False,
        "container_opening_radius": 0.035,
    },
    "notebook": {
        "graspable": True,
        "is_container": False,
        "is_surface": True,
    },
    "stapler": {
        "graspable": True,
        "is_container": False,
        "is_surface": False,
    },
    "candle": {
        "graspable": True,
        "is_container": False,
        "is_surface": False,
    },
    "plush_toy": {
        "graspable": True,
        "is_container": False,
        "is_surface": False,
    },
    # --- Non-graspable environment objects ---
    "table": {
        "graspable": False,
        "is_container": False,
        "is_surface": True,
    },
    "countertop": {
        "graspable": False,
        "is_container": False,
        "is_surface": True,
    },
}

# Default for unknown categories
_DEFAULT_AFFORDANCE = {
    "graspable": True,
    "is_container": False,
    "is_surface": False,
    "container_opening_radius": 0.0,
}


def infer_affordances(obj: SceneObject) -> SceneObject:
    """Populate affordance fields on a SceneObject based on its category.

    Also computes valid_as_target and valid_as_object task lists.
    Modifies the object in-place and returns it.
    """
    entry = AFFORDANCE_DB.get(obj.category, _DEFAULT_AFFORDANCE)

    obj.graspable = entry.get("graspable", True)
    obj.is_container = entry.get("is_container", False)
    obj.is_surface = entry.get("is_surface", False)
    obj.container_opening_radius = entry.get("container_opening_radius", 0.0)

    # --- What tasks can this object be the MANIPULATED object? ---
    obj.valid_as_object = []
    if obj.graspable:
        obj.valid_as_object.append(TaskType.PICK)
        # Graspable objects can be placed somewhere
        obj.valid_as_object.append(TaskType.PLACE_ON)
        obj.valid_as_object.append(TaskType.PLACE_IN)
        obj.valid_as_object.append(TaskType.PLACE_NEXT_TO)
        obj.valid_as_object.append(TaskType.STACK_ON)
    # Any object can be pushed
    obj.valid_as_object.append(TaskType.PUSH)

    # --- What tasks can this object be the TARGET? ---
    obj.valid_as_target = []
    if obj.is_surface:
        obj.valid_as_target.append(TaskType.PLACE_ON)
        obj.valid_as_target.append(TaskType.STACK_ON)
    if obj.is_container:
        obj.valid_as_target.append(TaskType.PLACE_IN)
    # Any object can be a spatial reference
    obj.valid_as_target.append(TaskType.PLACE_NEXT_TO)

    return obj
