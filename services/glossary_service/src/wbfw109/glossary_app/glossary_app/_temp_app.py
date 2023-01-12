"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import pynecone as pc
from pcconfig import config

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"


# class State(pc.State):
#     count: int = 0

#     def increment(self):
#         self.count += 1

#     def decrement(self):
#         self.count -= 1


# def index():
#     return pc.Hstack(
#         pc.button(
#             "Decrement",
#             color_scheme="red",
#             border_radius="1em",
#             on_click=State.decrement,
#         ),
#         pc.heading(State.count, font_size="2em"),
#         pc.button(
#             "Increment",
#             color_scheme="green",
#             border_radius="1em",
#             on_click=State.increment,
#         ),
#     )
class ExampleState(pc.State):
    """The app state."""

    # The colors to cycle through.
    colors = ["black", "red", "green", "blue", "purple"]

    # The index of the current color.
    index = 0

    def next_color(self):
        """Cycle to the next color."""
        self.index = (self.index + 1) % len(self.colors)

    @pc.var
    def color(self):
        return self.colors[self.index]


def get_example1() -> pc.Component:
    return pc.Hstack.create(
        pc.CircularProgress.create(
            pc.CircularProgressLabel.create("50", color="green"),
            value=50,
        ),
        pc.CircularProgress.create(
            pc.CircularProgressLabel.create("∞", color="rgb(107,99,246)"),
            is_indeterminate=True,
        ),
    )


def get_example2() -> pc.Component:
    return pc.center(
        pc.avatar(
            name="John Doe",
        )
    )


def index() -> pc.Component:
    return pc.center(
        pc.Vstack.create(
            pc.heading("Welcome to Pynecone!", font_size="2em"),
            pc.box("Get started by editing ", pc.code(filename, font_size="1em")),
            pc.link(
                "Check out our docs!",
                href=docs_url,
                border="0.1em solid",
                padding="0.5em",
                border_radius="0.5em",
                _hover={
                    "color": "rgb(107,99,246)",
                },
            ),
            spacing="1.5em",
            font_size="2em",
        ),
        padding_top="10%",
    )


def foo():
    return pc.heading(
        "Welcome to Pynecone!",
        on_click=ExampleState.next_color,
        color=ExampleState.color,
        _hover={"cursor": "pointer"},
    )


# Add state and page to the app.
app = pc.App(state=ExampleState)
app.add_page(index)
app.add_page(get_example2, path="/example/page", title="About Avatar page")
app.add_page(foo)
app.compile()

# ???  can only one State .. 오류 뭐지.. 근데 여기는 또 아님: https://github.com/pynecone-io/pynecone-examples/blob/main/twitter/twitter/twitter.py
# --------------------------------------------


class UppercaseState(pc.State):
    text: str = "hello"

    @pc.var
    def upper_text(self) -> str:
        return self.text.upper()

    def index():
        return pc.vstack(
            pc.heading(UppercaseState.upper_text),
            pc.input(
                on_blur=UppercaseState.set_text,
                placeholder="Type here...",
            ),
        )


class TickerState(pc.State):
    ticker: str = "AAPL"
    price: str = "$150"


def index():
    return pc.stat_group(
        pc.stat(
            pc.stat_label(TickerState.ticker),
            pc.stat_number(TickerState.price),
            pc.stat_help_text(
                pc.stat_arrow(type_="increase"),
                "4%",
            ),
        ),
    )


coins = ["BTC", "ETH", "LTC", "DOGE"]


class VarSelectState(pc.State):
    selected: str = "LTC"


def index():
    return pc.vstack(
        pc.heading("I just bought a bunch of " + VarSelectState.selected),
        pc.select(
            coins,
            on_change=VarSelectState.set_selected,
        ),
    )


class WordCycleState(pc.State):
    # The words to cycle through.
    text = ["Welcome", "to", "Pynecone", "!"]

    # The index of the current word.
    index = 0

    def next_word(self):
        self.index = (self.index + 1) % len(self.text)

    @pc.var
    def get_text(self):
        return self.text[self.index]


def index():
    return pc.heading(
        WordCycleState.get_text,
        on_mouse_over=WordCycleState.next_word,
        color="green",
    )


class ArgState(pc.State):
    colors: list[str] = [
        "rgba(222,44,12)",
        "white",
        "#007ac2",
    ]

    def change_color(self, color, index):
        self.colors[index] = color
        # Colors must be set not mutated (See warning below.)
        self.colors = self.colors


def index():
    return pc.hstack(
        pc.input(
            default_value=ArgState.colors[0],
            on_blur=lambda c: ArgState.change_color(c, 0),
            bg=ArgState.colors[0],
        ),
        pc.input(
            default_value=ArgState.colors[1],
            on_blur=lambda c: ArgState.change_color(c, 1),
            bg=ArgState.colors[1],
        ),
        pc.input(
            default_value=ArgState.colors[2],
            on_blur=lambda c: ArgState.change_color(c, 2),
            bg=ArgState.colors[2],
        ),
    )


options = ["1", "2", "3", "4"]


class SetterState2(pc.State):
    selected: str = "1"


def index():
    return pc.vstack(
        pc.badge(SetterState2.selected, color_scheme="green"),
        pc.select(
            options,
            on_change=SetterState2.set_selected,
        ),
    )


import asyncio


class ChainExampleState(pc.State):
    count = 0
    show_progress = False

    def toggle_progress(self):
        self.show_progress = not self.show_progress

    async def increment(self):
        # Think really hard.
        await asyncio.sleep(0.5)
        self.count += 1


def index():
    return pc.vstack(
        pc.badge(SetterState2.selected, color_scheme="green"),
        pc.select(
            options,
            on_change=SetterState2.set_selected,
        ),
    )


class CollatzState(pc.State):
    count: int = 0

    def start_collatz(self, count):
        """Run the collatz conjecture on the given number."""
        self.count = abs(int(count))
        return self.run_step

    async def run_step(self):
        """Run a single step of the collatz conjecture."""
        await asyncio.sleep(0.2)

        if self.count % 2 == 0:
            # If the number is even, divide by 2.
            self.count /= 2
        else:
            # If the number is odd, multiply by 3 and add 1.
            self.count = self.count * 3 + 1
        if self.count > 1:
            # Keep running until we reach 1.
            return self.run_step


def index():
    return pc.vstack(
        pc.badge(
            CollatzState.count,
            font_size="1.5em",
            color_scheme="green",
        ),
        pc.input(on_blur=CollatzState.start_collatz),
    )


class ServerSideState2(pc.State):
    def alert(self):
        return pc.window_alert("Hello World!")


def index():
    return pc.button("Alert", on_click=ServerSideState2.alert)
