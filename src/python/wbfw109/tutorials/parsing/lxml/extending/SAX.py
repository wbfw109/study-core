import lxml.sax
from io import BytesIO
from xml.sax.handler import ContentHandler
from xml.dom.pulldom import SAX2DOM

# isDebuggingTutorial = [0, 1, 2, 3, 4, 5, 6]
isDebuggingTutorial1 = [0, 0]
isDebuggingTutorial2 = [0, 0]
isDebuggingTutorial3 = [0, 0]


def SAX():
    ##########* ~~~ 1. Building a tree from SAX events ~~~
    handler = lxml.sax.ElementTreeContentHandler()
    handler.startElementNS((None, "root"), "", {})
    handler.startElementNS((None, "title"), "title", {(None, "foo"): "bar"})
    handler.characters("Hello world")
    handler.endElementNS((None, "title"), "title")
    handler.endElementNS((None, "root"), "")
    if isDebuggingTutorial1[1]:
        print(
            lxml.etree.tostring(handler.etree.getroot(), pretty_print=True).decode(
                "unicode_escape"
            )
        )
    # ??? By passing a makeelement function the constructor of ElementTreeContentHandler, e.g. the one of a parser you configured, you can determine which element class lookup scheme should be used.

    ##########* ~~~ 2. Producing SAX events from an ElementTree or Element ~~~
    f = BytesIO(b"<a><b>Text</b></a>")
    tree = lxml.etree.parse(f)
    """
    To see whether the correct SAX events are produced, we'll write a custom content handler. Note that it only defines the startElementNS() method and not startElement(). The SAX event generator in lxml.sax currently only supports namespace-aware processing.
  """

    class MyContentHandler(ContentHandler):
        def __init__(self):
            self.a_amount = 0
            self.b_amount = 0
            self.text = None

        def startElementNS(self, name, qname, attributes):
            uri, localname = name
            if localname == "a":
                self.a_amount += 1
            if localname == "b":
                self.b_amount += 1

        def characters(self, data):
            self.text = data

    handler = MyContentHandler()
    lxml.sax.saxify(tree, handler)
    if isDebuggingTutorial2[1]:
        print(handler.a_amount, handler.b_amount, handler.text)

    ##########* ~~~  3. Interfacing with pulldom/minidom ~~~
    # ???
    """"
    lxml.sax is a simple way to interface with the standard XML support in the Python library. Note, however, that this is a one-way solution, as Python's DOM implementation cannot generate SAX events from a DOM tree.
    You can use xml.dom.pulldom to build a minidom from lxml:
    PullDOM makes the result available through the document attribute:
  """
    handler = SAX2DOM()
    lxml.sax.saxify(tree, handler)
    # ??? .document property? where is docs in python.org/3 for this?
    dom = handler.document
    if isDebuggingTutorial3[1]:
        print(dom.firstChild.localName)


SAX()
"""
With Namespaces, elements and attributes have two-part name, sometimes called the "Universal" or "Expanded" name, which consists of a URI (signifying something analagous to a Java or Perl package name) and a localName (which never contains a colon).
"""
