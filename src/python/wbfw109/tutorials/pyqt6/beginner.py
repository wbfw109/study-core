import traceback
import sys
from typing import overload
from enum import Enum
import time
from PyQt6.QtWidgets import (
    QMenu,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QListWidget,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QStackedLayout,
    QTabWidget,
    QToolBar,
    QStatusBar,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
)
from PyQt6.QtCore import (
    QSize,
    Qt,
    QTimer,
    QRunnable,
    QThreadPool,
    pyqtSlot,
    QObject,
    pyqtSignal,
    pyqtBoundSignal,
    QProcess,
)
from PyQt6.QtGui import QAction, QPalette, QColor, QIcon, QKeySequence
from random import randint

# The core of every Qt Applications is the QApplication class. Every application needs one — and only one — QApplication object to function.


class MainWindowViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button_is_checked = True
        self.setWindowTitle("My App")
        self.button = QPushButton("Press Me!")
        # self.setFixedSize(QSize(1280, 720))
        self.resize(1280, 720)

        self.button.setCheckable(True)
        self.button.clicked.connect(self.the_button_was_clicked)
        self.button.clicked.connect(self.the_button_was_toggled)
        self.button.released.connect(self.the_button_was_released)
        # Set the central widget of the Window.
        self.setCentralWidget(self.button)

    def the_button_was_clicked(self):
        print("Clicked!")

    def the_button_was_toggled(self, checked):
        print("Checked?", checked)
        print(self.button_is_checked)
        # self.button.setText("You already clicked me.")
        # self.button.setEnabled(False)

    def the_button_was_released(self):
        self.button_is_checked = self.button.isChecked()

        print(self.button_is_checked)


class MainWindowWithLineEdit(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        self.label = QLabel()

        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)

        layout = QVBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)

        # self.setMouseTracking(True)
        # Set the central widget of the Window.
        self.setCentralWidget(container)

    def mouseMoveEvent(self, e):
        self.label.setText("mouseMoveEvent")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            # handle the left-button press in here
            self.label.setText("mousePressEvent LEFT")

        elif e.button() == Qt.MouseButton.MiddleButton:
            # handle the middle-button press in here.
            self.label.setText("mousePressEvent MIDDLE")

        elif e.button() == Qt.MouseButton.RightButton:
            # handle the right-button press in here.
            self.label.setText("mousePressEvent RIGHT")

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.label.setText("mouseReleaseEvent LEFT")

        elif e.button() == Qt.MouseButton.MiddleButton:
            self.label.setText("mouseReleaseEvent MIDDLE")

        elif e.button() == Qt.MouseButton.RightButton:
            self.label.setText("mouseReleaseEvent RIGHT")

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.label.setText("mouseDoubleClickEvent LEFT")

        elif e.button() == Qt.MouseButton.MiddleButton:
            self.label.setText("mouseDoubleClickEvent MIDDLE")

        elif e.button() == Qt.MouseButton.RightButton:
            self.label.setText("mouseDoubleClickEvent RIGHT")


class MainWindowContextMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.show()

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_context_menu)

    def on_context_menu(self, pos):
        context = QMenu(self)
        context.addAction(QAction("test 1", self))
        context.addAction(QAction("test 2", self))
        context.addAction(QAction("test 3", self))
        context.exec(self.mapToGlobal(pos))


# Subclass QMainWindow to customize your application's main window
class MainWindowManyWidgets(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Widgets App")

        layout = QVBoxLayout()
        widgets = [
            QCheckBox,
            QComboBox,
            QDateEdit,
            QDateTimeEdit,
            QDial,
            QDoubleSpinBox,
            QFontComboBox,
            QLCDNumber,
            QLabel,
            QLineEdit,
            QProgressBar,
            QPushButton,
            QRadioButton,
            QSlider,
            QSpinBox,
            QTimeEdit,
        ]

        for w in widgets:
            layout.addWidget(w())

        widget = QWidget()
        widget.setLayout(layout)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)


class EachQWidgets(Enum):
    QLabel = 1
    QCheckBox = 2
    QComboBox = 3
    QListWidget = 4
    QLineEdit = 5
    # QSpinBox, QDoubleSpinBox
    SpinBoxes = 6
    # QSlider, QDoubleSpinBox
    QSliders = 7
    QDial = 8


class MainWindowEachWidgets(QMainWindow):
    def __init__(self, oneOfEachQWidgets: EachQWidgets):
        super(MainWindowEachWidgets, self).__init__()
        self.setWindowTitle("My App")
        self.oneOfEachQWidgets = oneOfEachQWidgets

        match oneOfEachQWidgets:
            case EachQWidgets.QLabel:
                # widget.setPixmap(QPixmap('otje.jpg'))
                widget = QLabel("Hello")
                widget.setText("Hello2")
                font = widget.font()
                font.setPointSize(30)
                widget.setFont(font)
                widget.setAlignment(
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
                )
            case EachQWidgets.QCheckBox:
                widget = QCheckBox()
                widget.setCheckState(Qt.CheckState.Checked)
                # For tristate: widget.setCheckState(Qt.PartiallyChecked)
                widget.setTristate(True)
                widget.stateChanged.connect(self.show_state)
            case EachQWidgets.QComboBox:
                widget = QComboBox()
                # ??? widget.setInsertPolicy(QComboBox.InsertPolicy.InsertAtTop)
                widget.addItems(["One", "Two", "Three"])
                widget.setMaxCount(10)
                # widget.setEditable(True)
                # Sends the current index (position) of the selected item.
                widget.currentIndexChanged.connect(self.index_changed)
                # There is an alternate signal to send the text.
                widget.editTextChanged.connect(self.text_changed)
            case EachQWidgets.QListWidget:
                widget = QListWidget()
                widget.addItems(["One", "Two", "Three"])
                widget.currentItemChanged.connect(self.index_changed)
                widget.currentTextChanged.connect(self.text_changed)
            case EachQWidgets.QLineEdit:
                widget = QLineEdit()
                widget.setMaxLength(12)
                widget.setInputMask("000.000.000.000;_")
                widget.setPlaceholderText("Enter your text")

                # widget.setReadOnly(True) # uncomment this to make readonly

                widget.returnPressed.connect(self.return_pressed)
                widget.selectionChanged.connect(self.selection_changed)
                widget.textChanged.connect(self.text_changed)
                widget.textEdited.connect(self.text_edited)
            case EachQWidgets.SpinBoxes:
                widget = QSpinBox()
                # Or: widget = QDoubleSpinBox()

                widget.setMinimum(-10)
                widget.setMaximum(3)
                # Or: widget.setRange(-10,3)

                widget.setPrefix("$")
                widget.setSuffix("c")
                widget.setSingleStep(3)  # Or e.g. 0.5 for QDoubleSpinBox
                widget.valueChanged.connect(self.value_changed)
                widget.textChanged.connect(self.value_changed_str)
            case EachQWidgets.QSliders:
                widget = QSlider()
                widget.setMinimum(-10)
                widget.setMaximum(3)
                # Or: widget.setRange(-10,3)

                widget.setSingleStep(3)
                widget.setOrientation(Qt.Orientation.Horizontal)
                widget.valueChanged.connect(self.value_changed)
                widget.sliderMoved.connect(self.slider_position)
                widget.sliderPressed.connect(self.slider_pressed)
                widget.sliderReleased.connect(self.slider_released)
            case EachQWidgets.QDial:
                widget = QDial()
                widget.setRange(-10, 100)
                widget.setSingleStep(1)

                widget.valueChanged.connect(self.value_changed)
                widget.sliderMoved.connect(self.slider_position)
                widget.sliderPressed.connect(self.slider_pressed)
                widget.sliderReleased.connect(self.slider_released)
        self.setCentralWidget(widget)

    def show_state(self, s):
        print(s, s == Qt.CheckState.Checked.value)
        print(Qt.CheckState(s))

    def return_pressed(self):
        print("Return pressed!")
        self.centralWidget().setText("BOOM!")

    def selection_changed(self):
        print("Selection changed")
        print(self.centralWidget().selectedText())

    def index_changed(self, i):  # i is an int
        match self.oneOfEachQWidgets:
            case EachQWidgets.QComboBox:
                print(i)
            case EachQWidgets.QListWidget:
                print(i.text())

    def text_changed(self, s):  # s is a str
        print("Text changed...")
        print(s)

    def text_edited(self, s):
        print("Text edited...")
        print(s)

    def value_changed(self, i):
        print(i)

    def value_changed_str(self, s):
        print(s)

    def slider_position(self, p):
        print("position", p)

    def slider_pressed(self):
        print("Pressed!")

    def slider_released(self):
        print("Released")


class EachQBoxLayout(Enum):
    QHVBoxLayout = 1
    QGridLayout = 2
    QStackedLayout = 3
    QTabWidget = 4


class Color(QWidget):
    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)


class MainWindowLayout(QMainWindow):
    def __init__(self, oneOfEachQBoxLayout: EachQBoxLayout):
        super(MainWindowLayout, self).__init__()

        self.setWindowTitle("My App")

        match oneOfEachQBoxLayout:
            case EachQBoxLayout.QHVBoxLayout:
                layout1 = QHBoxLayout()
                layout2 = QVBoxLayout()
                layout3 = QVBoxLayout()

                layout1.setContentsMargins(0, 0, 0, 0)
                layout1.setSpacing(20)

                layout2.addWidget(Color("red"))
                layout2.addWidget(Color("yellow"))
                layout2.addWidget(Color("purple"))

                layout1.addLayout(layout2)

                layout1.addWidget(Color("green"))

                layout3.addWidget(Color("red"))
                layout3.addWidget(Color("purple"))

                layout1.addLayout(layout3)

                widget = QWidget()
                widget.setLayout(layout1)
                self.setCentralWidget(widget)
            case EachQBoxLayout.QGridLayout:
                layout = QGridLayout()

                layout.addWidget(Color("red"), 0, 0)
                layout.addWidget(Color("green"), 1, 0)
                layout.addWidget(Color("blue"), 1, 1)
                layout.addWidget(Color("purple"), 2, 1)

                widget = QWidget()
                widget.setLayout(layout)
                self.setCentralWidget(widget)
            case EachQBoxLayout.QStackedLayout:
                page_layout = QVBoxLayout()
                button_layout = QHBoxLayout()
                self.stackLayout = QStackedLayout()

                page_layout.addLayout(button_layout)
                page_layout.addLayout(self.stackLayout)

                btn = QPushButton("red")
                btn.pressed.connect(self.activate_tab_1)
                button_layout.addWidget(btn)
                self.stackLayout.addWidget(Color("red"))

                btn = QPushButton("green")
                btn.pressed.connect(self.activate_tab_2)
                button_layout.addWidget(btn)
                self.stackLayout.addWidget(Color("green"))

                btn = QPushButton("yellow")
                btn.pressed.connect(self.activate_tab_3)
                button_layout.addWidget(btn)
                self.stackLayout.addWidget(Color("yellow"))

                widget = QWidget()
                widget.setLayout(page_layout)
                self.setCentralWidget(widget)

            case EachQBoxLayout.QTabWidget:
                tabs = QTabWidget()
                tabs.setTabPosition(QTabWidget.TabPosition.West)
                tabs.setMovable(True)
                # tabs.setDocumentMode(True)

                for n, color in enumerate(["red", "green", "blue", "yellow"]):
                    tabs.addTab(Color(color), color)
                self.setCentralWidget(tabs)

    def activate_tab_1(self):
        self.stackLayout.setCurrentIndex(0)

    def activate_tab_2(self):
        self.stackLayout.setCurrentIndex(1)

    def activate_tab_3(self):
        self.stackLayout.setCurrentIndex(2)


class MainWindowToolbarsAndMenus(QMainWindow):
    # For this I recommend you download the fugue icon set by designer Yusuke Kamiyamane. http://p.yusukekamiyamane.com/
    # Note that Qt uses your operating system default settings to determine whether to show an icon, text or an icon and text in the toolbar. But you can override this by using .setToolButtonStyle. This slot accepts any of the following flags from the Qt. namespace:
    def __init__(self):
        super(MainWindowToolbarsAndMenus, self).__init__()

        self.setWindowTitle("My Awesome App")
        self.setStatusBar(QStatusBar(self))

        label = QLabel("Hello!")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(label)

        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        # button_action = QAction("Your button", self)
        button_action = QAction(
            QIcon("rsrc/fugue-icons-3.5.6/icons-shadowless/bug.png"),
            "Your button",
            self,
        )
        button_action.setStatusTip("This is your button")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        # There is also a .toggled signal, which only emits a signal when the button is toggled. But the effect is identical so it is mostly pointless.
        button_action.setCheckable(True)
        # Note that the keyboard shortcut is associated with the QAction and will still work whether or not the QAction is added to a menu or a toolbar.
        # Key sequences can be defined in multiple ways - either by passing as text, using key names from the Qt namespace, or using the defined key sequences from the Qt namespace. Use the latter wherever you can to ensure compliance with the operating system standards.
        # You can enter keyboard shortcuts using key names (e.g. Ctrl+p)
        # Qt.namespace identifiers (e.g. Qt.CTRL + Qt.Key_P)
        # or system agnostic identifiers (e.g. QKeySequence.Print)
        button_action.setShortcut(QKeySequence("Ctrl+p"))
        toolbar.addAction(button_action)

        toolbar.addSeparator()

        button_action2 = QAction(
            QIcon("rsrc/fugue-icons-3.5.6/icons-shadowless/bug.png"),
            "Your &button2",
            self,
        )
        button_action2.setStatusTip("This is your button2")
        button_action2.triggered.connect(self.onMyToolBarButtonClick)
        button_action2.setCheckable(True)
        toolbar.addAction(button_action2)

        toolbar.addWidget(QLabel("Hello"))
        toolbar.addWidget(QCheckBox())
        menu = self.menuBar()

        # We can reuse the already existing QAction to add the same function to the menu.
        # This won't be visible on macOS. Note that this is different to a keyboard shortcut — we'll cover that shortly.
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action)
        file_menu.addSeparator()
        file_submenu = file_menu.addMenu("Submenu")
        file_submenu.addAction(button_action2)

    def onMyToolBarButtonClick(self, s):
        print("click", s)


class EachQDialog(Enum):
    Default = 1
    QMessageBox = 2
    QMessageBoxWithStaticMethod = 3


class CustomDialogYesOrNo(QDialog):
    # You could of course choose to ignore this and use a standard QButton in a layout, but the approach outlined here ensures that your dialog respects the host desktop standards (OK on left vs. right for example).
    def __init__(self, parent=None):
        # When you click the button to launch the dialog, you may notice that it appears away from the parent window -- probably in the center of the screen. Normally you want dialogs to appear over their launching window to make them easier for users to find. To do this we need to give Qt a parent for the dialog.
        super().__init__(parent)

        self.setWindowTitle("HELLO!")

        QBtn = (
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel("Something happened, is that OK?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class MainWindowDialogs(QMainWindow):
    def __init__(self, oneOfEachQDialog: EachQDialog):
        super().__init__()

        self.setWindowTitle("My App")
        button = QPushButton("Press me for a dialog!")
        match oneOfEachQDialog:
            case EachQDialog.Default:
                button.clicked.connect(self.button_clicked)
            case EachQDialog.QMessageBox:
                button.clicked.connect(self.button_clicked_with_message_box)
            case EachQDialog.QMessageBoxWithStaticMethod:
                button.clicked.connect(
                    self.button_clicked_with_message_box_and_static_method
                )

        self.setCentralWidget(button)

    def button_clicked(self, s):
        print("click", s)

        dlg = CustomDialogYesOrNo(self)
        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")

    def button_clicked_with_message_box(self, s):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("I have a question!")

        # dlg.setText("This is a simple dialog")
        # button = dlg.exec()
        # if button == QMessageBox.StandardButton.Ok:
        #     print("OK!")

        dlg.setText("This is a question dialog")
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        dlg.setIcon(QMessageBox.Icon.Question)
        button = dlg.exec()

        if button == QMessageBox.StandardButton.Yes:
            print("Yes!")
        else:
            print("No!")

    def button_clicked_with_message_box_and_static_method(self, s):
        """
        The four information, question, warning and critical methods also accept optional buttons and defaultButton arguments which can be used to tweak the buttons shown on the dialog and select one by default.
            QMessageBox.about(parent, title, message)
            QMessageBox.critical(parent, title, message)
            QMessageBox.information(parent, title, message)
            QMessageBox.question(parent, title, message)
            QMessageBox.warning(parent, title, message)
        """
        button = QMessageBox.critical(
            self,
            "Oh dear!",
            "Something went very wrong.",
            buttons=QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.NoToAll
            | QMessageBox.StandardButton.Ignore,
            defaultButton=QMessageBox.StandardButton.Discard,
        )

        if button == QMessageBox.StandardButton.Discard:
            print("Discard!")
        elif button == QMessageBox.StandardButton.NoToAll:
            print("No to all!")
        else:
            print("Ignore!")


class EachWindows(Enum):
    Toggling = 1
    PersistentAndHiding = 2


class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Another Window % d" % randint(0, 100))
        layout.addWidget(self.label)
        self.setLayout(layout)


class MainWindowCreatingWindows(QMainWindow):
    def __init__(self, oneOfEachWindows: EachWindows):
        super().__init__()

        match oneOfEachWindows:
            case EachWindows.Toggling:
                self.w = None  # No external window yet.
                self.button = QPushButton("Push for Window")
                self.button.clicked.connect(self.show_new_window)
                self.setCentralWidget(self.button)
            case EachWindows.PersistentAndHiding:
                self.window1 = AnotherWindow()
                self.window2 = AnotherWindow()

                l = QVBoxLayout()
                button1 = QPushButton("Push for Window 1")
                button1.clicked.connect(self.toggle_window1)
                l.addWidget(button1)

                button2 = QPushButton("Push for Window 2")
                button2.clicked.connect(
                    lambda checked: self.toggle_window2(self.window2)
                )
                l.addWidget(button2)

                w = QWidget()
                w.setLayout(l)
                self.setCentralWidget(w)

    def show_new_window(self, checked):
        if self.w is None:
            self.w = AnotherWindow()
            self.w.show()
        else:
            self.w.close()  # Close window.
            # If we set it to  any other value that None the window will still close, but the if self.w is None test will not pass the next time we click the button and so we will not be able to recreate a window.
            # This will only work if you have not kept a reference to this window somewhere else. To make sure the window closes regardless, you may want to explicitly call .close() on it.
            self.w = None  # Discard reference, close window.

    def toggle_window1(self, checked):
        if self.window1.isVisible():
            self.window1.hide()

        else:
            self.window1.show()

    def toggle_window2(self, window: QMainWindow):
        if window.isVisible():
            window.hide()

        else:
            window.show()


class MainWindowAdvancedSignal1(QMainWindow):
    def __init__(self):
        super(MainWindowAdvancedSignal1, self).__init__()

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title will be passed to the function.
        self.windowTitleChanged.connect(self.on_window_title_changed)

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is discarded and the
        # function is called without parameters.
        self.windowTitleChanged.connect(
            lambda x: self.on_window_title_changed_no_params()
        )

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is discarded and the
        # function is called without parameters.
        # The function has default params.
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn())

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is passed to the function
        # and replaces the default parameter. Extra data is passed from
        # within the lambda.
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x, 25))

        # This sets the window title which will trigger all the above signals
        # sending the new title to the attached functions or lambdas as the
        # first parameter.
        self.setWindowTitle("My Signals App")

        checkbox = QCheckBox("Check?")

        # Option 1: conversion function
        def checkstate_to_bool(state: int):
            if Qt.CheckState(state) == Qt.CheckState.Checked:
                return self.result(True)

            return self.result(False)

        checkbox.stateChanged.connect(checkstate_to_bool)

        # Option 2: dictionary lookup
        checkbox.stateChanged.connect(lambda v: self.result(v))

        self.setCentralWidget(checkbox)

    # SLOT: This accepts a string, e.g. the window title, and prints it
    def on_window_title_changed(self, s):
        print(s)

    # SLOT: This is called when the window title changes.
    def on_window_title_changed_no_params(self):
        print("Window title changed.")

    # SLOT: This has default parameters and can be called without a value
    def my_custom_fn(self, a="HELLLO!", b=5):
        print(a, b)

    # SLOT: Accepts the check value.
    @overload
    def result(self, v: int) -> None:
        ...

    @overload
    def result(self, v: bool) -> None:
        ...

    def result(self, v):
        if type(v) is bool:
            print("bool:", v)
        elif type(v) is int:
            print("int:", v, Qt.CheckState(v))

    def button_pressed(self, n):
        self.label.setText(str(n))


class MainWindowAdvancedSignal2(QWidget):
    def __init__(self):
        super().__init__()

        v = QVBoxLayout()
        h = QHBoxLayout()

        for a in range(10):
            button = QPushButton(str(a))
            # The important thing is to use named parameters.
            button.pressed.connect(lambda x=a: self.button_pressed(x))
            h.addWidget(button)

        v.addLayout(h)
        self.label = QLabel("")
        v.addWidget(self.label)
        self.setLayout(v)

    def button_pressed(self, n):
        self.label.setText(str(n))


class MainWindowThreading1(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindowThreading1, self).__init__(*args, **kwargs)

        self.counter = 0

        layout = QVBoxLayout()

        self.l = QLabel("Start")
        b = QPushButton("DANGER!")
        b.pressed.connect(self.oh_no)

        layout.addWidget(self.l)
        layout.addWidget(b)

        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)

        self.show()

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def oh_no(self):
        # The simplest, and perhaps most logical, way to get around this is to accept events from within your code. This allows Qt to continue to respond to the host OS and your application will stay responsive
        # time.sleep(5)
        for n in range(5):
            """
            Note that Use of this function is discouraged. Instead, prefer to move long operations out of the GUI thread into an auxiliary one and to completely avoid nested event loop processing. If event processing is really necessary, consider using QEventLoop instead.
            Firstly, when you pass control back to Qt, your code is no longer running. This means that whatever long-running thing you're trying to do will take longer. That is definitely not what you want.
            Secondly, processing events outside the main event loop (app.exec_()) causes your application to branch off into handling code (e.g. for triggered slots, or events) while within your loop. If your code depends on/responds to external state this can cause undefined behavior. The code below demonstrates this in action:
            """
            QApplication.processEvents()
            time.sleep(1)

    def recurring_timer(self):
        self.counter += 1
        self.l.setText("Counter: %d" % self.counter)


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    You may not find a need for all of these signals, but they are included to give an indication of what is possible. In the following code we're going to implement a long-running task that makes use of these signals to provide useful information to the user.
    Supported signals are:
    - finished
        No data
    - error
        tuple (exctype, value, traceback.format_exc() )
    - result
        object data returned from processing, anything
    - progress
        int indicating % progress
    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    """
    Worker thread. Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    Executing our function in another thread is simply a matter of creating an instance of the Worker and then pass it to our QThreadPool instance and it will be executed automatically.
    If you want to pass custom data into the execution function you can do so via the init, and then have access to the data via self. from within the run slot.

    You also often want to receive status information from long-running threads. This can be done by passing in callbacks to which your running code can send the information. You have two options here: either define new signals (allowing the handling to be performed using the event loop) or use a standard Python function.
    In both cases you'll need to pass these callbacks into your target function to be able to use them. The signal-based approach is used in the completed code below, where we pass an int back as an indicator of the thread's % progress.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        # print("Thread start")
        # time.sleep(5)
        # print("Thread complete")

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class MainWindowThreading2(QMainWindow):
    def __init__(self, *args, **kwargs):
        """
        if you press the "?" button while oh_no is still running you'll see that the message changes. State is being changed from outside your loop.
        This is a toy example. However, if you have multiple long-running processes within your application, with each calling QApplication.processEvents() to keep things ticking, your application behaviour can be unpredictable.

        """
        super(MainWindowThreading2, self).__init__(*args, **kwargs)

        self.counter = 0
        layout = QVBoxLayout()

        self.l = QLabel("Start")
        b = QPushButton("DANGER!")
        b.pressed.connect(self.oh_no)

        layout.addWidget(self.l)
        layout.addWidget(b)

        # c = QPushButton("?")
        # c.pressed.connect(self.change_message)

        # layout.addWidget(c)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.show()

        self.threadpool = QThreadPool()
        print(
            "Multithreading with maximum %d threads" % self.threadpool.maxThreadCount()
        )

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    # def change_message(self):
    #     self.message = "OH NO"

    def progress_fn(self, n: int):
        print(f"{n}% done")

    def execute_this_fn(self, progress_callback: pyqtBoundSignal):
        # print("Hello!")
        for n in range(0, 5):
            time.sleep(1)
            progress_callback.emit(int(n * 100 / 4))
        return "Done."

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def oh_no(self):
        # self.message = "Pressed"
        # for n in range(100):
        #     time.sleep(0.1)
        #     self.l.setText(self.message)
        #     QApplication.processEvents()

        # Pass the function to execute
        worker = Worker(
            self.execute_this_fn
        )  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)

    def recurring_timer(self):
        self.counter += 1
        self.l.setText("Counter: %d" % self.counter)


class MainWindowMultiProcessing(QMainWindow):
    def __init__(self):
        # If you're running multiple external programs at once and do want to track their states, you may want to consider creating a manager class which does this for you
        super().__init__()

        self.p = None  # Default empty value.

        self.btn = QPushButton("Execute")
        self.btn.pressed.connect(self.start_process)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        l = QVBoxLayout()
        l.addWidget(self.btn)
        l.addWidget(self.text)

        w = QWidget()
        w.setLayout(l)

        self.setCentralWidget(w)

    def message(self, s):
        self.text.appendPlainText(s)
    
    def start_process(self):
        if self.p is None:  # No process running.
            self.message("Executing process")
            self.p = (
                QProcess()
            )  # Keep a reference to the QProcess (e.g. on self) while it's running.
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.finished.connect(self.process_finished)  # Clean up once complete.
            # If you run this example and press the button, nothing will happen. The external script is running but you can't see the output.
            self.p.start("python", ["wbfw109/tutorials/pyqt6/dummy_script.py"])

    def handle_stderr(self):
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.message(stderr)

    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.ProcessState.NotRunning: "Not running",
            QProcess.ProcessState.Starting: "Starting",
            QProcess.ProcessState.Running: "Running",
        }
        state_name = states[state]
        self.message(f"State changed: {state_name}")

    def process_finished(self):
        self.message("Process finished.")
        self.p = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # ~Getting started with PyQt6
    # window = MainWindowViewer()
    # window = MainWindowWithLineEdit()
    # window = MainWindowContextMenu()
    # window = MainWindowContextMenu()
    # window = MainWindowManyWidgets()
    # window = MainWindowEachWidgets(EachQWidgets.QDial)
    # window = MainWindowLayout(EachQBoxLayout.QTabWidget)
    # window = MainWindowToolbarsAndMenus()
    # window = MainWindowDialogs(EachQDialog.QMessageBoxWithStaticMethod)
    # window = MainWindowCreatingWindows(EachWindows.PersistentAndHiding)
    # ~Extended UI features
    # window = MainWindowAdvancedSignal1()
    # window = MainWindowAdvancedSignal2()
    # ~Threads & Processes
    # window = MainWindowThreading1()
    # window = MainWindowThreading2()

    window = MainWindowMultiProcessing()

    window.show()
    sys.exit(app.exec())


"""
CartonWorker

require Courses:
    Getting started with PyQt6
    Extended UI features
    Threads & Processes
    Packaging and distribution
"""
