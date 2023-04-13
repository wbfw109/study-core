"""
Discovered Bug (?)
    - When event: (pc.window_alert(self.message)) is registered in AccordionButton as double-click event trigger, only one time goes wells.
        ??? Microsoft Edge Browser problem? works well in private mode.
    - props <render_fn> of <pc.foreach> must return one Component.
        if you want to sibling component, surround these with component such as <pc.span> or use unpack character (*).
    - 🔧 pc.foreach can not process iterable data in a handler function.
        It is treated as pc.Base.Var, so you have to loop through classes or functions that don't inherit from pc.Base or pc.State.
            - Error example: AttributeError: 'tuple' object has no attribute 'id_'
        ; Functions in classes that inherit pc.State is Event handler. not common function.
        but, it can access by getitem "...[<key>]" (; Not works on nested TypedDict).
Tip
    - one of most important point is "Vars must be JSON serializable." 🔗 https://pynecone.io/docs/state/vars
        - ⚠️ It judge serializable from python typing (type hint) by using pydantic.  
            so it may occur error if a variable is serializable but type hint is wrong. 
    - variables in state class that inherits pc.State class are must be serializable.
        ; e.g. Using tuple key in dict is impossible.
        > TypeError: State vars must be primitive Python types, Plotly figures, Pandas dataframes, or subclasses of pc.Base. Found var "id_generator" with type typing.Iterator.
            pd.DataFrame example: NBA from https://pynecone.io/docs/gallery and https://pynecone.io/docs/library/datadisplay/datatable
    - If you use black formatter, some code will be wrapped as tuple.
        and if you use algorithms that returns nested component from nested call, note that unpack operator (*) may be required.
        📝 If that way is not consistent to these in a function, wrap it in a list and use the unpack operator (*) together.
    - If you want to pass argument that will be used in function to return pc.Component  
        , copy variables into pc.State.
        In this app, refer to class <CommonData> and class <WordDataState(pc.State)>.
    - If you want to class that inherits pd.Base as method argument in state that inherits pc.State  
        , it must be converted to json by using pd.Base method <~>.json() before pass.  
        otherwise TypeError occurs such as ```TypeError: Arguments to event handlers must be Vars or JSON-serializable. Got indexes=[] axis_index=None of type <class 'glossary_app.glossary_app.ToggleIndex'>.```
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from pprint import pprint
from typing import Any, ClassVar, Iterator, Optional, Self, TypedDict
from urllib.parse import quote

import pynecone as pc
from pcconfig import config

# Title: Load data ~
global_style: dict[Any, Any] = {"font_family": "Noto Emoji"}
anchor_with_sticky_header_compatible_style = {"scroll_margin_top": "5.5em"}
top_menu_bar_style: dict[Any, Any] = {
    "align_self": "flex-start",
    "position": "sticky",
    "top": "0",
    "z-index": "1",
    "width": "100%",
    "height": "3em",
    "background_color": "lightgray",
    "justify_contents": "space-between",
    "> *": {"margin": "4px 2px 2px 2px"},
}
one_level_sticky_body_section_flex_style: dict[Any, Any] = {
    "position": "sticky",
    "top": "2.9em",
    "z-index": "1",
    "width": "100%",
    "height": "100%",
    "justify_items": "stretch",
    "background": "lightgrey",
}
one_level_sticky_tos_flex_style: dict[Any, Any] = {
    "position": "sticky",
    "top": "-0.5em",
    "z-index": "1",
    "width": "100%",
    "height": "100%",
    "justify_items": "stretch",
    "background": "lightgrey",
}


class PyneconeFixedData:
    accordion_button_id_prefix: str = "accordion-button-"


def get_emoji_of_number(number: int, /) -> str:
    """
    Args:
        number (int): 0 ~ 9. if other value provided, just return the number as string.
    Returns:
        str: emoji according to number or just the number string.
    """
    if number < 0 or number > 9:
        return f"{number}"
    return ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"][
        number
    ]


class WordData(TypedDict):
    word_name: str
    children: list[WordData]
    description: list[str]


class WordDataPC(pc.Base):
    """If <parent_index_id> is -1, it indicates that a WordDataPC is child of root node."""

    index_id: int
    word_name: str
    description: list[str]


class WordDataSerializable:
    """
    Attributes:
        - self.word_data_pc_list: list[WordDataPC]
        - self.children_index_dict: dict[int, list[int]]
            key "-1" is pseudo root.
        - self.word_name_list_for_id: list[str]


    It processes tasks:
        - serialize word data into object <WordDataPC(pc.Base)> form from nested shape.
        - create dict: <parent_index> to <children_index> of word data.

    Implementation:
        - Index starts from 0 because Accordion component uses index props to toggle collapse Accordion components.
    """

    START_INDEX_ID: ClassVar[int] = 0
    MIN_DEPTH_LEVEL: ClassVar[int] = 1
    ROOT_PSEUDO_INDEX_ID: ClassVar[int] = -1

    def __init__(self, word_data_path: Optional[Path] = None) -> None:
        self.count_iterator: Iterator[int] = itertools.count(0)

        # endpoint of method <_get_word_data_pc>
        if not word_data_path:
            self.word_data_path: Path = Path(
                "../../../../../ref/computer_science_words_korean.json"
            )
        else:
            self.word_data_path = word_data_path
        with self.word_data_path.open("r", encoding="UTF-8") as f:
            word_data_list: list[WordData] = json.load(f)

        # following attributes will be set in method <_get_word_data_pc>
        self.word_name_list_for_id: list[str] = []
        self.word_href_list_for_id: list[str] = []
        self.children_indexes_dict: dict[int, list[int]] = {
            WordDataSerializable.ROOT_PSEUDO_INDEX_ID: []
        }
        self.ascendant_indexes_dict: dict[int, list[int]] = {}
        self.descendant_indexes_dict: dict[int, list[int]] = {}
        self.word_data_pc_list: list[WordDataPC] = self._get_word_data_pc_and_set_meta(
            word_data_list=word_data_list
        )

    def _get_word_data_pc_and_set_meta(
        self,
        word_data_list: list[WordData],
        parent_index_id: int = ROOT_PSEUDO_INDEX_ID,
        depth_level: int = MIN_DEPTH_LEVEL,
        ascendant_indexes: Optional[list[int]] = None,
    ) -> list[WordDataPC]:
        """It iteratively parses list of WordData, that is nested dict.

        It will set values following defined attributes:
        ```
        self.word_name_list_for_id: list[str] = []
        self.word_href_list_for_id: list[str] = []
        self.children_index_dict: dict[int, list[int]] = {
            WordDataSerializable.ROOT_PSEUDO_INDEX_ID: []
        }
        self.ascendants_index_dict: dict[int, list[int]] = {}
        self.descendant_indexes_dict: dict[int, list[int]] = {}
        ```

        ---

        Implementation:
            - Entry point target is list in json.
            - It uses Depth First Search and process in order (precedence of left node rather than right node) if same depth level.
                if a node's children does not exists return the self node tree with descendants to the parent node.
        """
        word_data_pc_list: list[WordDataPC] = []
        for word_data in word_data_list:
            # print(type(word_data["word_name"]))
            # Title: pre process
            if not ascendant_indexes:
                ascendant_indexes = []
            # get next id from Generator
            index_id: int = next(self.count_iterator)

            # Title: process
            self.children_indexes_dict[index_id] = []
            self.children_indexes_dict[parent_index_id].append(index_id)
            if type(word_data) == str:
                print(word_data)

            self.word_name_list_for_id.append(word_data["word_name"])
            self.word_href_list_for_id.append(
                f"#{PyneconeFixedData.accordion_button_id_prefix}"
                + quote(word_data["word_name"])
            )
            word_data_pc_list.append(
                WordDataPC(
                    index_id=index_id,
                    word_name=f"{get_emoji_of_number(depth_level)} {word_data['word_name']}",
                    description=word_data["description"],
                )
            )

            word_data_pc_len_before_explore: int = len(word_data_pc_list)
            if word_data["children"]:
                # if children node exists, explore nested node firstly.
                word_data_pc_list.extend(
                    self._get_word_data_pc_and_set_meta(
                        word_data_list=word_data["children"],
                        parent_index_id=index_id,
                        depth_level=depth_level + 1,
                        ascendant_indexes=ascendant_indexes + [index_id],
                    )
                )
            else:
                # if leaf node, set ascendants of the leaf node into <self.ascendants_index_dict>.
                # And trace next parent and iteratively process.
                self.ascendant_indexes_dict[index_id] = ascendant_indexes
                for count, i in enumerate(reversed(ascendant_indexes), start=1):
                    if i in self.ascendant_indexes_dict:
                        break
                    self.ascendant_indexes_dict[i] = ascendant_indexes[:-count]

            # set <self.descendant_indexes_dict> from descendants node tree of current node.
            word_data_pc_len_after_explore: int = len(word_data_pc_list)
            self.descendant_indexes_dict[index_id] = [
                word_data_pc.index_id
                for word_data_pc in word_data_pc_list[
                    word_data_pc_len_before_explore : word_data_pc_len_after_explore + 1
                ]
            ]

        return word_data_pc_list


class CommonData:
    WORD_DATA_SERIALIZABLE: WordDataSerializable = WordDataSerializable()
    WORD_DATA_PC_LIST: list[WordDataPC] = WORD_DATA_SERIALIZABLE.word_data_pc_list
    WORD_DATA_CHILDREN_INDEXES_DICT: dict[
        int, list[int]
    ] = WORD_DATA_SERIALIZABLE.children_indexes_dict
    WORD_DATA_DESCENDANT_INDEXES_DICT: dict[
        int, list[int]
    ] = WORD_DATA_SERIALIZABLE.descendant_indexes_dict
    WORD_DATA_ASCENDANT_INDEXES_DICT: dict[
        int, list[int]
    ] = WORD_DATA_SERIALIZABLE.ascendant_indexes_dict
    WORD_NAME_LIST_FOR_ID: list[str] = WORD_DATA_SERIALIZABLE.word_name_list_for_id
    WORD_HREF_LIST_FOR_ID: list[str] = WORD_DATA_SERIALIZABLE.word_href_list_for_id
    DEFAULT_INDEX: list[int] = list(range(len(WORD_DATA_PC_LIST)))
    DEFAULT_IS_EXPANDED_LIST: list[bool] = [True for _ in range(len(DEFAULT_INDEX))]
    # in dev set 1.0, in prod set 0.5 due to lag
    MOUSE_UP_EVENT_TRIGGER_TIME: float = 0.5


# Title: State ~


class ExpandableAccordion(pc.Base):
    expanded_index_list: list[int]
    is_expanded_list: list[bool]

    @staticmethod
    def get_default_object() -> Self:
        return ExpandableAccordion(
            expanded_index_list=CommonData.DEFAULT_INDEX.copy(),
            is_expanded_list=CommonData.DEFAULT_IS_EXPANDED_LIST.copy(),
        )


class WordDataState(pc.State):
    """- mouse-up event occurs faster than mouse-click event"""

    # ? Type ExpandableAccordion is not serializable...? when modularize...
    WORD_DATA_PC_LIST: list[WordDataPC] = CommonData.WORD_DATA_PC_LIST
    WORD_DATA_ASCENDANT_INDEXES_DICT: dict[
        int, list[int]
    ] = CommonData.WORD_DATA_ASCENDANT_INDEXES_DICT
    WORD_HREF_LIST_FOR_ID: list[str] = CommonData.WORD_HREF_LIST_FOR_ID
    side_bar: ExpandableAccordion = ExpandableAccordion.get_default_object()
    body_section: ExpandableAccordion = ExpandableAccordion.get_default_object()
    mouse_down_start_time: float = 0.0

    # top menu bar
    shows_left_drawer: bool = False

    def toggle_left_drawer(self):
        self.shows_left_drawer = not self.shows_left_drawer

    def toggle_side_bar_accordion_item_descendants(
        self, axis_index: Optional[int]
    ) -> None:
        value_to_be_set: bool = True

        if not axis_index:
            if len(self.side_bar.expanded_index_list) == len(CommonData.DEFAULT_INDEX):
                value_to_be_set = False
            self.side_bar.is_expanded_list = [
                value_to_be_set for _ in range(len(CommonData.DEFAULT_INDEX))
            ]
        else:
            # current bug: "Argument type is converted in Event chain"
            axis_index = int(axis_index)
            descendants_index: list[int] = CommonData.WORD_DATA_DESCENDANT_INDEXES_DICT[
                axis_index
            ]
            if not descendants_index:
                return
            else:
                value_to_be_set = not self.side_bar.is_expanded_list[
                    descendants_index[0]
                ]

            for i in descendants_index:
                self.side_bar.is_expanded_list[i] = value_to_be_set

        self.side_bar.expanded_index_list = list(
            itertools.compress(
                CommonData.DEFAULT_INDEX,
                self.side_bar.is_expanded_list,
            )
        )
        self.side_bar = ExpandableAccordion(
            is_expanded_list=self.side_bar.is_expanded_list,
            expanded_index_list=self.side_bar.expanded_index_list,
        )

    def toggle_body_section_accordion_item_descendants(
        self, axis_index: Optional[int]
    ) -> None:
        value_to_be_set: bool = True

        if not axis_index:
            if len(self.body_section.expanded_index_list) == len(
                CommonData.DEFAULT_INDEX
            ):
                value_to_be_set = False
            self.body_section.is_expanded_list = [
                value_to_be_set for _ in range(len(CommonData.DEFAULT_INDEX))
            ]
        else:
            # current bug: "Argument type is converted in Event chain"
            axis_index = int(axis_index)
            descendants_index: list[int] = CommonData.WORD_DATA_DESCENDANT_INDEXES_DICT[
                axis_index
            ]
            if not descendants_index:
                return
            else:
                value_to_be_set = not self.body_section.is_expanded_list[
                    descendants_index[0]
                ]

            for i in descendants_index:
                self.body_section.is_expanded_list[i] = value_to_be_set

        self.body_section.expanded_index_list = list(
            itertools.compress(
                CommonData.DEFAULT_INDEX,
                self.body_section.is_expanded_list,
            )
        )
        self.body_section = ExpandableAccordion(
            is_expanded_list=self.body_section.is_expanded_list,
            expanded_index_list=self.body_section.expanded_index_list,
        )

    def move_to_permalink(self, axis_index: int) -> None:
        """DOING: Link component href have precedence than on click event.. so.. how..."""
        # close drawer
        self.shows_left_drawer = not self.shows_left_drawer

        # fold <axis_index> and ascendants of <axis_index>.
        self.body_section.is_expanded_list[axis_index] = True
        for i in CommonData.WORD_DATA_ASCENDANT_INDEXES_DICT[axis_index]:
            self.body_section.is_expanded_list[i] = True

        self.body_section.expanded_index_list = list(
            itertools.compress(
                CommonData.DEFAULT_INDEX,
                self.body_section.is_expanded_list,
            )
        )
        self.body_section = ExpandableAccordion(
            is_expanded_list=self.body_section.is_expanded_list,
            expanded_index_list=self.body_section.expanded_index_list,
        )

        # move to permalink
        # return pc.redirect(f"/{CommonData.WORD_HREF_LIST_FOR_ID[axis_index]}")

    def toggle_side_bar_accordion_item(self, index: int) -> None:
        if (
            time.time() - self.mouse_down_start_time
        ) >= CommonData.MOUSE_UP_EVENT_TRIGGER_TIME:
            return self.toggle_side_bar_accordion_item_descendants(index)

        self.side_bar.is_expanded_list[index] = not self.side_bar.is_expanded_list[
            index
        ]
        self.side_bar.expanded_index_list = list(
            itertools.compress(
                CommonData.DEFAULT_INDEX,
                self.side_bar.is_expanded_list,
            )
        )
        self.side_bar = ExpandableAccordion(
            is_expanded_list=self.side_bar.is_expanded_list,
            expanded_index_list=self.side_bar.expanded_index_list,
        )

    def toggle_body_section_accordion_item(self, index: int) -> None:
        if (
            time.time() - self.mouse_down_start_time
        ) >= CommonData.MOUSE_UP_EVENT_TRIGGER_TIME:
            return self.toggle_body_section_accordion_item_descendants(index)

        index = int(index)
        self.body_section.is_expanded_list[
            index
        ] = not self.body_section.is_expanded_list[index]

        self.body_section.expanded_index_list = list(
            itertools.compress(
                CommonData.DEFAULT_INDEX,
                self.body_section.is_expanded_list,
            )
        )
        self.body_section = ExpandableAccordion(
            is_expanded_list=self.body_section.is_expanded_list,
            expanded_index_list=self.body_section.expanded_index_list,
        )

    def mouse_down_event(self) -> None:
        self.mouse_down_start_time = time.time()

    def update_search_result(self, input_text: str) -> None:
        if len(input_text.strip()) <= 1:
            return


# Title: Components ~
def wrap_page_tooltip(component: pc.component):
    return pc.tooltip(
        component,
        label="📖 click element: Toggle, 📚 long click element >= 0.5 second: Toggle descendants.",
    )


def get_icon_that_move_to_body_section(index: int):
    """DOING: Link component href have precedence than on click event.. so.. how..."""
    return pc.link(
        pc.tooltip(
            pc.button(
                pc.icon(tag="InfoOutlineIcon"),
                bg="whitesmoke",
                on_click=lambda x: WordDataState.move_to_permalink(index),
            ),
            label="Move to body section",
        ),
        rel="help",
        href=CommonData.WORD_HREF_LIST_FOR_ID[index],
    )


def get_icon_self_link_in_body_section(index: int):
    return pc.link(
        pc.tooltip(
            pc.button(
                pc.icon(tag="LinkIcon"),
                bg="whitesmoke",
            ),
            label="Permalink",
        ),
        rel="bookmark",
        href=CommonData.WORD_HREF_LIST_FOR_ID[index],
    )


def get_icon_that_shows_breadcrumb(index: int):
    return pc.menu(
        pc.tooltip(
            pc.MenuButton.create(
                pc.button(
                    pc.icon(tag="TriangleUpIcon"),
                    bg="floralwhite",
                ),
            ),
            label="Breadcrumb",
        ),
        pc.MenuList.create(
            pc.foreach(
                WordDataState.WORD_DATA_ASCENDANT_INDEXES_DICT[index],
                lambda index: pc.MenuItem.create(
                    pc.link(
                        WordDataState.WORD_DATA_PC_LIST[index].word_name,
                        rel="bookmark",
                        href=WordDataState.WORD_HREF_LIST_FOR_ID[index],
                    )
                ),
            )
        ),
        is_lazy=True,
        lazy_behavior="keepMounted",
    )


def get_word_accordion_by_depth(
    word_data_pc_list: list[WordDataPC],
    current_index: int = WordDataSerializable.ROOT_PSEUDO_INDEX_ID,
) -> list[pc.Accordion]:
    """It returns word description as well as word name."""
    # <next_parent_key> is key of current word node.
    nested_components: list = []

    # current index will be next parent index.
    for child_key in CommonData.WORD_DATA_CHILDREN_INDEXES_DICT[current_index]:
        nested_components.append(
            *get_word_accordion_by_depth(
                word_data_pc_list,
                current_index=child_key,
            )
        )

    if current_index == WordDataSerializable.ROOT_PSEUDO_INDEX_ID:
        # if root element
        return pc.Accordion.create(
            *nested_components,
            allow_multiple=True,
            reduce_motion=True,
            default_index=[],
            index=WordDataState.body_section.expanded_index_list,
            width="100%",
        )
    else:
        current_word_data = word_data_pc_list[current_index]
        current_style: dict[str, str] = {}
        if current_index in CommonData.WORD_DATA_CHILDREN_INDEXES_DICT[-1]:
            current_style = one_level_sticky_body_section_flex_style

        return [
            pc.AccordionItem.create(
                pc.flex(
                    get_icon_self_link_in_body_section(current_index),
                    get_icon_that_shows_breadcrumb(current_index),
                    pc.AccordionButton.create(
                        pc.text(current_word_data.word_name),
                        pc.AccordionIcon.create(),
                        on_click=lambda x: WordDataState.toggle_body_section_accordion_item(
                            current_index
                        ),
                        on_mouse_down=WordDataState.mouse_down_event,
                        style=anchor_with_sticky_header_compatible_style,
                    ),
                    style=current_style,
                ),
                pc.AccordionPanel.create(
                    pc.foreach(
                        current_word_data.description,
                        lambda line: pc.text(line),
                    ),
                    *nested_components,
                ),
                id_=CommonData.WORD_NAME_LIST_FOR_ID[current_index],
            )
        ]


def get_word_tos_accordion_by_depth(
    word_data_pc_list: list[WordDataPC],
    current_index: int = WordDataSerializable.ROOT_PSEUDO_INDEX_ID,
) -> list[pc.Accordion]:
    # <next_parent_key> is key of current word node.
    nested_components: list = []

    # current index will be next parent index.
    for child_key in CommonData.WORD_DATA_CHILDREN_INDEXES_DICT[current_index]:
        nested_components.append(
            *get_word_tos_accordion_by_depth(word_data_pc_list, current_index=child_key)
        )

    if current_index == WordDataSerializable.ROOT_PSEUDO_INDEX_ID:
        # if root element
        return pc.Accordion.create(
            *nested_components,
            allow_multiple=True,
            reduce_motion=True,
            default_index=WordDataState.side_bar.expanded_index_list,
            index=WordDataState.side_bar.expanded_index_list,
            width="100%",
        )
    else:
        current_word_data = word_data_pc_list[current_index]
        if nested_components:
            current_style: dict[str, str] = {}
            if current_index in CommonData.WORD_DATA_CHILDREN_INDEXES_DICT[-1]:
                current_style = one_level_sticky_tos_flex_style
            return [
                pc.AccordionItem.create(
                    pc.flex(
                        get_icon_that_move_to_body_section(current_index),
                        pc.AccordionButton.create(
                            pc.text(current_word_data.word_name),
                            pc.AccordionIcon.create(),
                            on_click=lambda x: WordDataState.toggle_side_bar_accordion_item(
                                current_index
                            ),
                            on_mouse_down=WordDataState.mouse_down_event,
                        ),
                        style=current_style,
                    ),
                    pc.AccordionPanel.create(
                        *nested_components,
                    ),
                )
            ]
        else:
            # if leaf node
            return [
                pc.AccordionItem.create(
                    pc.flex(
                        get_icon_that_move_to_body_section(current_index),
                        pc.AccordionButton.create(
                            pc.text(current_word_data.word_name),
                        ),
                    )
                ),
            ]


# Title: Routing ~


def index() -> pc.Component:
    return pc.flex(
        # Top Menu Bar ~
        pc.flex(
            pc.button(
                pc.icon(tag="HamburgerIcon"),
                bg="lightblue",
                on_click=WordDataState.toggle_left_drawer,
            ),
            # Side bar ~
            pc.drawer(
                pc.DrawerOverlay.create(
                    pc.DrawerContent.create(
                        pc.DrawerHeader.create(
                            wrap_page_tooltip(
                                pc.text(
                                    "TOS",
                                    background_image="linear-gradient(271.68deg, #EE756A 0.75%, #756AEE 88.52%)",
                                    background_clip="text",
                                    font_weight="bold",
                                    font_size="2em",
                                )
                            ),
                            pc.divider(),
                            pc.button(
                                "Toggle all Fold",
                                on_click=lambda x: WordDataState.toggle_side_bar_accordion_item_descendants(
                                    None
                                ),
                                color_scheme="green",
                                variant="outline",
                                border_radius="1em",
                                margin_top="0.5em",
                            ),
                        ),
                        pc.DrawerBody.create(
                            pc.flex(
                                get_word_tos_accordion_by_depth(
                                    WordDataState.WORD_DATA_PC_LIST
                                ),
                            )
                        ),
                        pc.DrawerFooter.create(
                            pc.button(
                                "Close", on_click=WordDataState.toggle_left_drawer
                            )
                        ),
                    )
                ),
                placement="left",
                size="lg",
                is_open=WordDataState.shows_left_drawer,
                on_close=WordDataState.toggle_left_drawer,
                close_on_esc=True,
            ),
            # Search bar ~
            pc.button(
                pc.icon(tag="Search2Icon"),
                bg="white",
                margin_left="1em",
                _hover={"cursor": "default"},
            ),
            pc.input(
                # Type something...
                placeholder="🚩 Not yet implemented overlay search bar",
                bg="white",
                width="100%",
                margin_right="0.5em",
                on_change=lambda text: WordDataState.update_search_result(text),
            ),
            style=top_menu_bar_style,
        ),
        pc.divider(border_color="black", margin="5px"),
        # Body Section ~
        pc.flex(
            wrap_page_tooltip(
                pc.text(
                    "Contents",
                    style={
                        "color": "green",
                        "font_size": "2em",
                        "font_weight": "bold",
                        "box_shadow": "rgba(240, 46, 170, 0.4) 5px 5px, rgba(240, 46, 170, 0.3) 10px 10px",
                    },
                )
            ),
            pc.divider(),
            pc.button(
                "Toggle all Fold",
                on_click=lambda x: WordDataState.toggle_body_section_accordion_item_descendants(
                    None
                ),
                color_scheme="green",
                variant="outline",
                border_radius="1em",
                margin_top="0.5em",
                margin_bottom="0.5em",
            ),
            get_word_accordion_by_depth(WordDataState.WORD_DATA_PC_LIST),
            direction="column",
            align_items="flex-start",
            justify_items="stretch",
        ),
        direction="column",
    )


app = pc.App(state=WordDataState, style=global_style)
app.add_page(index, title="Glossary App")
app.compile()
