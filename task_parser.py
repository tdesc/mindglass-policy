"""Grammar for the RoboLab/MolmoSpaces task catalog (id-style and sentence-style).

Pure stdlib: no numpy/cv2/positronic imports, so it is importable in any Python >= 3.9
(including verification venvs that lack the robot stack).

Catalog ids look like ``pick-spoon-onto-table``, ``move-cup-left``, ``open-the-drawer``,
``close-the-drawer``, ``put-fork-on-plate``, ``spoon-from-bowl-to-cup``,
``bottle-in-the-bin`` — i.e. hyphen- or space-joined tokens
``[verb] [the] [color] object [relation target]``. Sentences from the sim
(``pick up the red block and place it in the box``) follow the same token grammar.

Output of :func:`parse_task_text` (the contract with scripted_policy):

    {
      'skill': 'pick' | 'place' | 'open_drawer' | 'close_drawer' | 'move_dir' | 'give_up',
      'object_color_or_type': <color word if the task names one, else object noun, else None>,
      'target': <normalized relation, see TARGET docs below, or None>,
      'verb': <raw leading verb or None>,
      'object': <canonical object noun or None>,
      'color': <color word or None>,
      'raw': <original text>,
    }

``target`` values the FSM understands:
  - ``'onto-<noun>'``   put the object onto a surface/thing (table, plate, pot, ...)
  - ``'into-<noun>'``   release above/inside a container (bin, box, basket, bowl, cup, ...)
  - ``'left'/'right'``  carry the object laterally (screen direction; world axis unknown)
  - ``'left-of:<noun>' / 'right-of:<noun>' / 'beside:<noun>'``
  - ``'from:<noun>'``   pick it out of that location (pick-only)
  - ``'from-to:<a>:<b>'`` pick from a, place at b (``spoon-from-bowl-to-cup``)
  - ``'drawer-open' / 'drawer-close'`` for the drawer skill

``skill == 'give_up'`` means the text is outside the grammar (unknown leading verb,
garbage, or a recognized-but-unsupported verb such as ``pour``): the FSM ends the
episode honestly instead of guessing.
"""

from __future__ import annotations

import re
from typing import Any

SKILL_PICK = 'pick'
SKILL_PLACE = 'place'
SKILL_OPEN_DRAWER = 'open_drawer'
SKILL_CLOSE_DRAWER = 'close_drawer'
SKILL_MOVE_DIR = 'move_dir'
SKILL_GIVE_UP = 'give_up'

# Verbs the FSM can actually execute, plus synonyms seen in catalogs/sentences.
PICK_VERBS = {'pick', 'grab', 'take', 'get', 'lift'}
PLACE_VERBS = {'put', 'place', 'set', 'drop', 'lay'}
MOVE_VERBS = {'move', 'push', 'slide', 'shift', 'carry', 'drag'}
DRAWER_OPEN_VERBS = {'open'}
DRAWER_CLOSE_VERBS = {'close', 'shut'}
UNSUPPORTED_VERBS = {'pour'}  # recognized, but no pouring skill: honest give_up
VERB_MAP: dict[str, str] = {}
for _v in PICK_VERBS:
    VERB_MAP[_v] = 'pick'
for _v in PLACE_VERBS:
    VERB_MAP[_v] = 'place'
for _v in MOVE_VERBS:
    VERB_MAP[_v] = 'move'
for _v in DRAWER_OPEN_VERBS:
    VERB_MAP[_v] = 'open'
for _v in DRAWER_CLOSE_VERBS:
    VERB_MAP[_v] = 'close'
for _v in UNSUPPORTED_VERBS:
    VERB_MAP[_v] = _v

# Object dictionary (canonical names). Synonyms map into canon via OBJECT_SYNONYMS.
# Furniture/appliance nouns are needed so open/close tasks keep their real target object
# (submission-3 log: 'open the dresser/refrigerator/oven/dishwasher/cabinet/fridge') —
# scripted_policy classifies the open pull direction (slide vs hinged door) from it.
OBJECTS = {
    'spoon', 'fork', 'knife', 'cup', 'cube', 'box', 'bin', 'basket', 'bowl', 'plate',
    'pot', 'pan', 'bottle', 'marker', 'pen', 'banana', 'apple', 'towel', 'drawer',
    'lid', 'ball', 'can', 'sponge', 'cheezit', 'mustard', 'ketchup', 'toy', 'chip',
    'cubelet', 'table', 'counter', 'desk', 'surface', 'shelf',
    'dresser', 'cabinet', 'cupboard', 'door', 'oven', 'dishwasher', 'refrigerator',
    'wardrobe', 'freezer', 'microwave',
}
OBJECT_SYNONYMS = {
    'mug': 'cup', 'glass': 'cup', 'block': 'cube', 'pencil': 'pen', 'cloth': 'towel',
    'bandage': 'towel', 'container': 'box', 'trash': 'bin', 'garbage': 'bin',
    'tray': 'plate', 'skillet': 'pan', 'cracker': 'cheezit', 'fruit': 'apple',
    'fridge': 'refrigerator',
}
# Nouns the FSM may release into/from above (place target 'into-<noun>').
CONTAINERS = {'box', 'bin', 'basket', 'bowl', 'cup', 'pot', 'pan', 'drawer', 'can'}
# Surfaces/calm targets ('onto-<noun>', blind table height).
SURFACES = {'table', 'counter', 'plate', 'tray', 'pot', 'pan', 'desk', 'surface', 'shelf'}

# Colour words. The first four are the vision-supported HSV set (scripted_policy.COLOR_RANGES);
# the rest parse as colours but the FSM treats them as "no known colour" (largest-blob search).
VISION_COLORS = ('red', 'green', 'blue', 'yellow')
OTHER_COLORS = ('orange', 'purple', 'pink', 'black', 'white', 'brown', 'gray', 'grey')
COLORS = set(VISION_COLORS) | set(OTHER_COLORS)

# Words that never mean anything by themselves: articles, pronouns, particles,
# size/position modifiers, verb particles (pick **up**, set **down**). The material/
# appearance adjectives ('wooden', 'dark', 'tall', …) come from the real catalog texts
# (submission-3 log: 'open the tall dark refrigerator', 'close the light brown table') —
# without them _noun_at stops before the noun and open/close tasks lose their object.
_SKIP = {
    'the', 'a', 'an', 'it', 'its', 'them', 'that', 'this', 'there', 'up', 'down',
    'wide', 'top', 'bottom', 'upper', 'lower', 'middle', 'small', 'big', 'large',
    'tiny', 'near', 'far', 'over', 'above', 'under', 'please', 'and', 'then',
    'slowly', 'gently', 'again', 'here', 'away', 'back', 'out', 'of', 'to', 'from',
    'onto', 'into', 'inside', 'within', 'on', 'in', 'at', 'by', 'with', 'beside',
    'next', 'left', 'right', 'of', 'for', 'off',
    'wooden', 'dark', 'light', 'tall', 'metal', 'plastic', 'narrow', 'long', 'short',
}


def singular(token: str) -> str:
    """Naive singular form; the catalog uses plural nouns sometimes (``stack-the-cubes``)."""
    if token in OBJECTS or token in OBJECT_SYNONYMS:
        return token
    for candidate in (token[:-1], token[:-2], token[:-3] + 'y') if token.endswith('s') else ():
        if candidate in OBJECTS or candidate in OBJECT_SYNONYMS:
            return candidate
    return token


def canon_object(token: str) -> str | None:
    """Canonical object noun for a token, or None."""
    token = singular(token)
    if token in OBJECTS:
        return token
    return OBJECT_SYNONYMS.get(token)


def _noun_at(tokens: list[str], i: int) -> tuple[str | None, int]:
    """Object noun at/after index ``i`` skipping articles/modifiers; returns (noun, next_i)."""
    j = i
    while j < len(tokens):
        token = tokens[j]
        noun = canon_object(token)
        if noun is not None:
            return noun, j + 1
        if token in _SKIP or token in COLORS:
            j += 1
            continue
        return None, j + 1
    return None, len(tokens)


def parse_task_text(task: str) -> dict[str, Any]:
    """Parse one catalog task id / sentence into the FSM intent dict (module docstring)."""
    raw = task or ''
    tokens = re.findall(r'[a-z]+', raw.lower())
    base: dict[str, Any] = {
        'skill': SKILL_GIVE_UP, 'object_color_or_type': None, 'target': None,
        'verb': None, 'object': None, 'color': None, 'raw': raw,
    }
    if not tokens:
        return base

    # -- leading verb (or its honest absence) ---------------------------------
    verb: str | None = None
    start = 0
    if tokens[0] in VERB_MAP:
        verb = VERB_MAP[tokens[0]]
        start = 1
    elif tokens[0] in _SKIP or canon_object(tokens[0]) is not None or tokens[0] in COLORS:
        pass  # object-first id ('spoon-from-bowl-to-cup') or an article — fine
    else:
        return base  # unknown leading word ('dance', 'stack', 'unscrew') → give_up

    # -- object and colour ----------------------------------------------------
    object_name, after_object = _noun_at(tokens, start)
    color = next((t for t in tokens if t in COLORS), None)
    if object_name is None and verb in DRAWER_OPEN_VERBS | DRAWER_CLOSE_VERBS:
        object_name = 'drawer'  # 'open the top drawer' with 'drawer' swallowed by a modifier

    # -- relations ------------------------------------------------------------
    direction: str | None = None
    left_of = right_of = beside = into = onto = from_noun = to_noun = None
    i = start
    while i < len(tokens):
        token = tokens[i]
        if token == 'left' or token == 'right':
            noun, _ = _noun_at(tokens, i + 1)
            if noun is not None:
                if token == 'left':
                    left_of = left_of or noun
                else:
                    right_of = right_of or noun
                i += 1
                continue
            direction = direction or token
        elif token in ('into', 'inside', 'within') or (token == 'in' and 'to' not in tokens[i + 1:i + 2]):
            noun, _ = _noun_at(tokens, i + 1)
            into = into or noun
        elif token in ('onto', 'upon') or (token == 'on' and tokens[i + 1:i + 2] != ['to']):
            noun, _ = _noun_at(tokens, i + 1)
            onto = onto or noun
        elif token == 'beside' or (token == 'next' and tokens[i + 1:i + 2] == ['to']):
            noun, _ = _noun_at(tokens, i + (2 if token == 'next' else 1))
            beside = beside or noun
            i += 1 if token == 'next' else 0  # 'next' consumed 'to' as well; skip the extra step
        elif token == 'from':
            noun, _ = _noun_at(tokens, i + 1)
            if noun is not None:
                from_noun = from_noun or noun
        elif token == 'out' and tokens[i + 1:i + 2] == ['of']:
            noun, _ = _noun_at(tokens, i + 2)
            if noun is not None:
                from_noun = from_noun or noun
        elif token == 'to':
            # _noun_at skips 'the/...'; bare directions ('to the left') yield None here and
            # are caught by the bare 'left'/'right' branch on a later token instead.
            noun, _ = _noun_at(tokens, i + 1)
            if noun is not None:
                to_noun = to_noun or noun
        i += 1  # every branch either 'continue'd past consumed tokens or falls through here
    if to_noun is not None and from_noun is not None:
        target = f'from-to:{from_noun}:{to_noun}'
    elif to_noun is not None:
        target = f'into-{to_noun}' if to_noun in CONTAINERS else f'onto-{to_noun}'
    elif into is not None:
        target = f'into-{into}'
    elif onto is not None:
        target = f'onto-{onto}'
    elif left_of is not None:
        target = f'left-of:{left_of}'
    elif right_of is not None:
        target = f'right-of:{right_of}'
    elif beside is not None:
        target = f'beside:{beside}'
    elif direction is not None:
        target = direction
    elif from_noun is not None:
        target = f'from:{from_noun}'
    else:
        target = None
    # Normalize surface/container spellings of onto/into targets.
    if isinstance(target, str) and target.startswith('onto-'):
        noun = target[5:]
        if noun in ('counter', 'desk', 'surface', 'shelf'):
            target = 'onto-table'
        elif noun == 'tray':
            target = 'onto-plate'

    # -- colour normalization for the vision layer -----------------------------
    color_out = color if color in VISION_COLORS else None
    object_color_or_type = color_out or object_name

    # -- skill decision --------------------------------------------------------
    if verb in DRAWER_OPEN_VERBS:
        skill = SKILL_OPEN_DRAWER
        target = 'drawer-open'
    elif verb in DRAWER_CLOSE_VERBS:
        skill = SKILL_CLOSE_DRAWER
        target = 'drawer-close'
    elif verb in UNSUPPORTED_VERBS:
        skill = SKILL_GIVE_UP  # recognized, honestly unsupported
    else:
        place_like = target is not None and (
            target.startswith(('onto-', 'into-')) or target.startswith(('left-of:', 'right-of:', 'beside:'))
            or target.startswith('from-to:')
        )
        if verb == 'pick':
            if place_like:
                skill = SKILL_PLACE
            elif direction is not None:
                skill = SKILL_MOVE_DIR
                target = direction
            else:
                skill = SKILL_PICK  # target 'from:<noun>' or none: pick it up
        elif verb == 'place':
            if direction is not None:
                skill = SKILL_MOVE_DIR
                target = direction
            else:
                skill = SKILL_PLACE
        elif verb == 'move':
            if direction is not None:
                skill = SKILL_MOVE_DIR
                target = direction
            elif place_like:
                skill = SKILL_PLACE
            else:
                skill = SKILL_GIVE_UP  # 'move the cup' with no direction: honest give_up
        elif object_name is not None:  # verb-less catalog ids
            if direction is not None:
                skill = SKILL_MOVE_DIR
                target = direction
            elif place_like:
                skill = SKILL_PLACE
            else:
                skill = SKILL_PICK
        else:
            skill = SKILL_GIVE_UP

    return {
        'skill': skill,
        'object_color_or_type': object_color_or_type if skill != SKILL_GIVE_UP else None,
        'target': target if skill != SKILL_GIVE_UP else None,
        'verb': verb,
        'object': object_name,
        'color': color_out,
        'raw': raw,
    }
