from lxml import etree
from lxml.builder import E, ElementMaker
import codecs
from io import BytesIO

# isDebuggingTutorial = [0, 1, 2, 3, 4, 5, 6]
isDebuggingTutorial1 = [0, 0, 0, 0, 0, 0, 0]
isDebuggingTutorial2 = [0, 0]
isDebuggingTutorial3 = [0, 0, 0, 0, 0, 0, 0]
isDebuggingTutorial4 = [0, 0]
isDebuggingTutorial5 = [0, 0]
isDebuggingTutorial6 = [0, 0]
# ??? element 를 보내면 마지막에 None 이 출력된다.
# def printElement(element):
#     print (etree.tostring(element, pretty_print=True).decode("unicode_escape"))
"""Note
    Use end of tutorial function about 1. ElementClass Element, 2. ElementTreeClass
"""


def tutorial():
  ##########* ~~~ 1. ElementClass ~~~
  ###* [1-1. Elements are lists]
  root = etree.Element("root", interesting="totally")
  root.append(etree.Element("child1"))
  """this is so common that there is a shorter and much more efficient way to do this: the SubElement factory. It accepts the same arguments as the Element factory, but additionally requires the parent as first argument:
  """
  child2 = etree.SubElement(root, "child2")
  child3 = etree.SubElement(root, "child3")
  root.insert(0, etree.Element("child0"))
  child = root[0]
  """In this example, the last element is moved to a different position, instead of being copied, i.e. it is automatically removed from its previous position when it is put in a different place. If you want to copy an element to a different position in lxml.etree, consider creating an independent deep copy using the copy module from Python's standard library:
  """
  # root[0] = root[-1]
  if isDebuggingTutorial1[1]:
    print("\n  1. ----------------------------------------\n")
    print(etree.iselement(root))

  ###* [1-2. Elements carry attributes as a dict]
  root.set("hello", "Huhu")
  rootAttributes = root.attrib
  # * To get an independent snapshot of the attributes that does not depend on the XML tree, copy it into a dict:  ->  Memory management ?
  if isDebuggingTutorial1[2]:
    print("\n  2. ----------------------------------------\n")
    if len(root):
      print(child.tag, len(root), root.index(root[1]))
    print(root[0].tag, root[-1].tag)
    print(
        root is root[0].getparent(),
        root[0] is root[1].getprevious(),
        root[1] is root[0].getnext(),
    )
    print(root.get("interesting"), root.get("hello"), root.keys())
    for name, value in sorted(root.items()):
      print("%s = %r" % (name, value))

  ###* [1-3. Elements contain text]
  # * if text of element is not None, \n character between subElements is disapeeared.
  child.text = "TEXT"
  child2.text = "Knowledge"
  """refer to site about "tail" property referred to as document-style or mixed-content XML. However, there are cases where the tail text also gets in the way. For example, when you serialize an Element from within the tree, you do not always want its tail text in the result (although you would still want the tail text of its children). For this purpose, the tostring() function accepts the keyword argument with_tail.
    If you want to read only the text, i.e. without any intermediate tags, you have to recursively concatenate all text and tail attributes in the correct order. Again, the tostring() function comes to the rescue, this time using the method keyword:
  """
  if isDebuggingTutorial1[3]:
    print("\n  3. ----------------------------------------\n")

  ###* [1-4. Using XPath to find text]
  build_text_list = etree.XPath("//text()")
  # If you want to use this more often, you can wrap it in a function:
  """Note that a string result returned by XPath is a special 'smart' object that knows about its origins. You can ask it where it came from through its getparent() method, just as you would with Elements.
    While this works for the results of the text() function, lxml will not tell you the origin of a string value that was constructed by the XPath functions string() or concat().  ->  None
  """
  parentXpathElement = build_text_list(child)[0]
  if isDebuggingTutorial1[4]:
    print("\n  4. ----------------------------------------\n")
    print(
        parentXpathElement.getparent().tag,
        parentXpathElement.is_text,
        parentXpathElement.is_tail,
    )
    print(root.xpath("string()"), root.xpath("//text()"), sep="  :  ")

  ###* [1-5. Tree iteration]
  etree.SubElement(root, "another").text = "Child 4"
  """If you know you are only interested in a single tag, you can pass its name to iter() to have it filter for you. Starting with lxml 3.0, you can also pass more than one tag to intercept on multiple tags during iteration.
  """
  child3.append(etree.Entity("#234"))
  child3.append(etree.Comment("some comment"))
  """
    By default, iteration yields all nodes in the tree, including ProcessingInstructions, Comments and Entity instances. If you want to make sure only Element objects are returned, you can pass the Element factory as tag parameter.
    Note that passing a wildcard "*" tag name will also yield all Element nodes (and only elements). In lxml.etree, elements provide further iterators for all directions in the tree: children, parents (or rather ancestors) and siblings.
  """
  if isDebuggingTutorial1[5]:
    print("\n  5. ----------------------------------------\n")
    for element in root.iter():
      print("%s - %s" % (element.tag, element.text))
    print()
    for element in root.iter("child0"):
      print("child0: %s - %s" % (element.tag, element.text))
    print()
    for element in root.iter("another", "child2"):
      print("another, child2: %s - %s" % (element.tag, element.text))
    print()
    for element in root.iter():
      if isinstance(element.tag, str):
        print("isInstance: %s - %s" % (element.tag, element.text))
      else:
        print("SPECIAL: %s - %s" % (element, element.text))
    print()
    for element in root.iter(tag=etree.Element):
      print("%s - %s" % (element.tag, element.text))

  ###* [1-6. Serialization]
  """Note that pretty printing appends a newline at the end. For more fine-grained control over the pretty-printing, you can add whitespace indentation to the tree before serialising it, using the indent() function (added in lxml 4.5).
    In lxml 2.0 and later (as well as ElementTree 1.3), the serialization functions can do more than XML serialization. You can serialize to HTML or extract the text content by passing the method keyword. default of method argument is XML.
    As for XML serialization, the default encoding for plain text serialization is ASCII. The default encoding for Python source code is UTF-
  """
  serializationTest1 = etree.tostring(root, method="text")
  serializationTest2 = etree.tostring(root, encoding="unicode", method="text")
  serializationTest3 = etree.tostring(root, method="XML")
  serializationTest4 = etree.tostring(root, method="HTML")
  if isDebuggingTutorial1[6]:
    print("\n  6. ----------------------------------------\n")
    print(
        serializationTest1,
        serializationTest2,
        serializationTest3,
        serializationTest4,
        sep="\n",
    )

  ##########* ~~~ 2. ElementTreeClass ~~~
  """
    An ElementTree is mainly a document wrapper around a tree with a root node. It provides a couple of methods for serialization and general document handling. An ElementTree is also what you get back when you call the parse() function to parse files or file-like objects (see the parsing section below).
    One of the important differences is that the ElementTree class serializes as a complete document, as opposed to a single Element. This includes top-level processing instructions and comments, as well as a DOCTYPE and other DTD content in the document:

  """
  tree = etree.ElementTree(root)
  tree.docinfo.public_id = "-//W3C//DTD XHTML 1.0 Transitional//EN"
  tree.docinfo.system_url = "file://local.dtd"
  ## AttributeError: attribute 'xml_version' of 'lxml.etree.DocInfo' objects is not writable
  # tree.docinfo.xml_version = '1.0'
  if isDebuggingTutorial2[1]:
    print("\n  7. ----------------------------------------\n")
    print(tree.docinfo.xml_version)
    print(tree.docinfo.doctype)
  """In the original xml.etree.ElementTree implementation and in lxml up to 1.3.3, the output looks the same as when serializing only the root Element. This serialization behavior has changed in lxml 1.3.4. Before, the tree was serialized without DTD content, which made lxml lose DTD information in an input-output cycle.
        print(etree.tostring(tree.getroot()))
      """

  ##########* ~~~ 3. Parsing from strings and files ~~~
  ###* [3-1. The fromstring() function]
  """lxml.etree supports parsing XML in a number of ways and from all important sources, namely strings, files, URLs (http/ftp) and file-like objects.
  """
  some_xml_data = "<root>data<child1>c1</child1></root>"
  rootFromSomeDataA = etree.fromstring(some_xml_data)
  if isDebuggingTutorial3[1]:
    print(rootFromSomeDataA.tag, rootFromSomeDataA[0].tag)

  ###* [3-2. The XML() function]
  """The XML() function behaves like the fromstring() function, but is commonly used to write XML literals right into the source:
  """
  rootFromSomeDataB = etree.XML(some_xml_data)
  if isDebuggingTutorial3[2]:
    print(rootFromSomeDataB.tag, rootFromSomeDataB[0].tag)

  ###* [3-3. The parse() function]
  """The parse() function is used to parse from files and file-like objects.
    As an example of such a file-like object, the following code uses the BytesIO class for reading from a string instead of an external file. That class comes from the io module in Python 2.6 and later. In older Python versions, you will have to use the StringIO class from the StringIO module.
  """
  # * However, in real life, you would obviously avoid doing this all together and use the string parsing functions above.
  some_file_or_file_like_object = BytesIO(b"<root>data</root>")
  rootFromSomeDataC = etree.parse(some_file_or_file_like_object)
  """The reasoning behind this difference is that parse() returns a complete document from a file, while the string parsing functions are commonly used to parse XML fragments.
    The parse() function supports any of the following sources:
        - an open file object (make sure to open it in binary mode)
        - a file-like object that has a .read(byte_count) method returning a byte string on each call
        - a filename string
        - an HTTP or FTP URL string
  """
  # * Note that passing a filename or URL is usually faster than passing an open file or file-like object. However, the HTTP/FTP client in libxml2 is rather simple, so things like HTTP authentication require a dedicated URL request library, e.g. urllib2 or requests. These libraries usually provide a file-like object for the result that you can parse from while the response is streaming in.
  if isDebuggingTutorial3[3]:
    print(rootFromSomeDataC.getroot())
    for element in rootFromSomeDataC.iter():
      print("%s - %s" % (element.tag, element.text))

  ###* [3-4. The Parser objects]
  """By default, lxml.etree uses a standard parser with a default setup. If you want to configure the parser, you can create a new instance
  """
  parserD = etree.XMLParser(remove_blank_text=True)
  # * This creates a parser that removes empty text between tags while parsing, which can reduce the size of the tree and avoid dangling tail text if you know that whitespace-only content is not meaningful for your data. An example:
  rootFromSomeDataD = etree.XML("<root>  <a/>   <b>  </b>     </root>", parserD)
  """
    Note that the whitespace content inside the <b> tag was not removed, as content at leaf elements tends to be data content (even if blank). You can easily remove it in an additional step by traversing the tree:
  """
  for element in rootFromSomeDataD.iter("*"):
    if element.text is not None and not element.text.strip():
      element.text = None
  if isDebuggingTutorial3[4]:
    print(etree.tostring(rootFromSomeDataD))

  ###* [3-5. Incremental parsing]
  """lxml.etree provides two ways for incremental step-by-step parsing. One is through file-like objects, where it calls the read() method repeatedly. This is best used where the data arrives from a source like urllib or any other file-like object that can provide data on request. Note that the parser will block and wait until data becomes available in this case:
  """

  class DataSource:
    data = [b"<roo", b"t><", b"aaaaaa/", b"><", b"/root>"]

    def read(self, requested_size):
      try:
        return self.data.pop(0)
      except IndexError:
        return b""

  rootFromSomeDataEA = etree.parse(DataSource())
  """The second way is through a feed parser interface, given by the feed(data) and close() methods.
    Here, you can interrupt the parsing process at any time and continue it later on with another call to the feed() method. This comes in handy if you want to avoid blocking calls to the parser, e.g. in frameworks like Twisted, or whenever data comes in slowly or in chunks and you want to do other things while waiting for the next chunk.
  """
  parserE = etree.XMLParser()
  parserE.feed("<roo")
  parserE.feed("t><")
  parserE.feed("bbb/")
  parserE.feed("><")
  parserE.feed("/root>")
  rootFromSomeDataEB = parserE.close()
  if isDebuggingTutorial3[5]:
    print(
        etree.tostring(rootFromSomeDataEA),
        etree.tostring(rootFromSomeDataEB),
        sep="\n",
    )
  """After calling the close() method (or when an exception was raised by the parser), you can reuse the parser by calling its feed() method again:
  """
  parserE.feed("<root/>")
  rootFromSomeDataEB = parserE.close()
  if isDebuggingTutorial3[5]:
    print(etree.tostring(rootFromSomeDataEB))

  ###* [3-6. Event-driven parsing]
  """
    Sometimes, all you need from a document is a small fraction somewhere deep inside the tree, so parsing the whole tree into memory, traversing it and dropping it can be too much overhead. lxml.etree supports this use case with two event-driven parser interfaces, one that generates parser events while building the tree (iterparse), and one that does not build the tree at all, and instead calls feedback methods on a target object in a SAX-like fashion.
  """
  some_file_like = BytesIO(b"<root><a><b>data</b></a><a><b/></a></root>")
  rootFromSomeDataF = etree.parse(some_file_like)
  if isDebuggingTutorial3[6]:
    for event, element in etree.iterparse(some_file_like):
      # * Note that the text, tail, and children of an Element are not necessarily present yet when receiving the start event. Only the end event guarantees that the Element has been parsed completely.
      if element.tag == "b":
        print(element.text)
      elif element.tag == "a":
        print("** cleaning up the subtree")
        element.clear()
        # element.clear(keep_tail=True)
        # * It also allows you to .clear() or modify the content of an Element to save memory.
    print(
        etree.tostring(rootFromSomeDataF,
                       pretty_print=True).decode("unicode_escape"))
    print()

  # * A very important use case for iterparse() is parsing large generated XML files, e.g. database dumps. Most often, these XML formats only have one main data item element that hangs directly below the root node and that is repeated thousands of times. In this case, it is best practice to let lxml.etree do the tree building and only to intercept on exactly this one Element, using the normal tree API for data extraction.
  xml_file = BytesIO(b"""\
        <root>
            <a><b>ABC</b><c>abc</c></a>
            <a><b>MORE DATA</b><c>more data</c></a>
            <ab><b>XYZ</b><c>xyz</c></ab>
            <ab><b></b><c>zyx</c></ab>
        </root>""")
  if isDebuggingTutorial3[6]:
    for _, element in etree.iterparse(xml_file, tag="a"):
      print("%s -- %s" % (element.findtext("b"), element[1].text))
      element.clear(keep_tail=True)
    print("\n----- ★ SAX fashion -----\n")

  # *??? If, for some reason, building the tree is not desired at all, the target parser interface of lxml.etree can be used. It creates SAX-like events by calling the methods of a target object. By implementing some or all of these methods, you can control which events are generated.
  """
    You can reuse the parser and its target as often as you like, so you should take care that the .close() method really resets the target to a usable state (also in the case of an error!).
  """

  class ParserTarget:
    events = []
    close_count = 0

    def start(self, tag, attrib):
      self.events.append(("start", tag, attrib))

    def close(self):
      events, self.events = self.events, []
      self.close_count += 1
      return events

  parser_target = ParserTarget()
  parserF = etree.XMLParser(target=parser_target)
  events = etree.fromstring('<root test="true"/>', parserF)
  if isDebuggingTutorial3[6]:
    print(parser_target.close_count)
    for event in events:
      print("event: %s - tag: %s" % (event[0], event[1]))
      for attr, value in event[2].items():
        print(" * %s = %s" % (attr, value))
    print()
    for i in range(3):
      events = etree.fromstring('<root test="true"/>', parserF)
      print(parser_target.close_count)
      for event in events:
        print("event: %s - tag: %s" % (event[0], event[1]))
        for attr, value in event[2].items():
          print(" * %s = %s" % (attr, value))
    print()

  ##########* ~~~ 4. Namespaces ~~~
  # *The ElementTree API avoids namespace prefixes wherever possible and deploys the real namespace (the URI) instead. As you can see from the example, prefixes only become important when you serialize the result.
  xhtml = etree.Element("{http://www.w3.org/1999/xhtml}html")
  body = etree.SubElement(xhtml, "{http://www.w3.org/1999/xhtml}body")
  body.text = "Hello World"
  if isDebuggingTutorial4[1]:
    print(etree.tostring(xhtml, pretty_print=True).decode("unicode_escape"))
    print()
  # *However, the above code looks somewhat verbose due to the lengthy namespace names. And retyping or copying a string over and over again is error prone. It is therefore common practice to store a namespace URI in a global variable.To adapt the namespace prefixes for serialization, you can also pass a mapping to the Element factory function, e.g. to define the default namespace:
  XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
  XHTML = "{%s}" % XHTML_NAMESPACE
  NSMAP = {None: XHTML_NAMESPACE}  # the default namespace (no prefix)
  # ???
  xhtml = etree.Element(XHTML + "html", nsmap=NSMAP)
  body = etree.SubElement(xhtml, XHTML + "body")
  body.text = "Hello World"

  if isDebuggingTutorial4[1]:
    print(etree.tostring(xhtml, pretty_print=True).decode("unicode_escape"))
    print()
  """You can also use the QName helper class to build or split qualified tag names:
  """
  # * You can also use the QName helper class to build or split qualified tag names:
  tagA = etree.QName(XHTML_NAMESPACE, "html")
  tagB = etree.QName(XHTML + "html")
  tagC = etree.QName(xhtml)
  tagD = etree.QName(xhtml, "changed_tag")
  # Alternatively, pass an Element to extract its tag name. None as first argument is ignored in order to allow for generic 2-argument usage
  if isDebuggingTutorial4[1]:
    print("--- QName ---")
    print(tagA, tagA == tagB == tagC)
    print(tagA.localname, tagA.namespace, tagA.text, sep="  :  ")
    print(tagB.localname, tagB.namespace, tagB.text, sep="  :  ")
    print(tagC.localname, tagC.namespace, tagC.text, sep="  :  ")
    print(tagD.localname, tagD.namespace, tagD.text, sep="  :  ")
    print()
  # * lxml.etree allows you to look up the current namespaces defined for a node through the .nsmap property:Note, however, that this includes all prefixes known in the context of an Element, not only those that it defines itself.
  root = etree.Element("root", nsmap={"a": "http://a.b/c"})
  child = etree.SubElement(root, "child", nsmap={"b": "http://b.c/d"})
  if isDebuggingTutorial4[1]:
    print(xhtml.nsmap)
    print(len(root.nsmap), len(child.nsmap))
    print(child.nsmap["a"], child.nsmap["b"])
  # * Namespaces on attributes work alike, but as of version 2.3, lxml.etree will ensure that the attribute uses a prefixed namespace declaration. This is because unprefixed attribute names are not considered being in a namespace by the XML namespace specification (section 6.2), so they may end up losing their namespace on a serialise-parse roundtrip, even if they appear in a namespaced element.
  body.set(XHTML + "bgcolor", "#CCFFAA")
  if isDebuggingTutorial4[1]:
    print(etree.tostring(xhtml, pretty_print=True).decode("unicode_escape"))
    print(body.get("bgcolor"), body.get(XHTML + "bgcolor"))
    print()
  """You can also use XPath with fully qualified names.
  """
  find_xhtml_body = etree.ETXPath("//{%s}body" % XHTML_NAMESPACE)
  # ??? 사용법 나오지 않음..  인스턴스 이름으로 인수를 넣어서 결과값 도출?
  results = find_xhtml_body(xhtml)
  if isDebuggingTutorial4[1]:
    print("--- ETXPath; Element Tree XPath ---")
    print(results[0], results[0].tag)
    print()
  """For convenience, you can use "*" wildcards in all iterators of lxml.etree, both for tag names and namespaces:
    To look for elements that do not have a namespace, either use the plain tag name or provide the empty namespace explicitly
  """
  if isDebuggingTutorial4[1]:
    for el in xhtml.iter("*"):
      print(el.tag)  # any element
    print()
    for el in xhtml.iter("{http://www.w3.org/1999/xhtml}*"):
      print(el.tag)
    print()
    for el in xhtml.iter("{*}body"):
      print(el.tag)
    print()
    print([el.tag for el in xhtml.iter("body")])
    print([el.tag for el in xhtml.iter("{}body")])
    print([el.tag for el in xhtml.iter("{}*")])
    print()

  ##########* ~~~ 5. Namespaces ~~~
  """The E-factory provides a simple and compact syntax for generating XML and HTML:
  """

  def CLASS(*args):  # class is a reserved word in Python
    return {"class": " ".join(args)}

  html = page = E.html(  # create an Element called "html"
      E.head(E.title("This is a sample document")),
      E.body(
          E.h1("Hello!", CLASS("title")),
          # it convert {"class": 'title'} into <~ class="title">
          E.p("This is a paragraph with ", E.b("bold"), " text in it!"),
          E.p(
              "This is another paragraph, with a",
              "\n      ",
              E.a("link", href="http://www.python.org"),
              ".",
          ),
          E.p("Here are some reserved characters: <spam&egg>."),
          etree.XML("<p>And finally an embedded XHTML fragment.</p>"),
      ),
  )
  if isDebuggingTutorial5[1]:
    print(etree.tostring(html, pretty_print=True).decode("unicode_escape"))
    print()
  """Element creation based on attribute access makes it easy to build up a simple vocabulary for an XML language. When dealing with multiple namespaces, it is good practice to define one ElementMaker for each namespace URI. Again, note how the above example predefines the tag builders in named constants. That makes it easy to put all tag declarations of a namespace into one Python module and to import/use the tag name constants from there. This avoids pitfalls like typos or accidentally missing namespaces.
  """
  Em = ElementMaker(
      namespace="http://my.de/fault/namespace",
      nsmap={None: "http://my.de/fault/namespace"},
  )
  # nsmap={'p': "http://my.de/fault/namespace"}
  DOC = Em.doc
  TITLE = Em.title
  SECTION = Em.section
  PAR = Em.par
  my_doc = DOC(
      TITLE("The dog and the hog"),
      SECTION(TITLE("The dog"), PAR("Once upon a time, ..."),
              PAR("And then ...")),
      SECTION(TITLE("The hog"), PAR("Sooner or later ...")),
  )
  if isDebuggingTutorial5[1]:
    print(etree.tostring(my_doc, pretty_print=True).decode("unicode_escape"))

  ##########* ~~~ 6. ElementPath ~~~
  # * The ElementTree library comes with a simple XPath-like path language called ElementPath. The main difference is that you can use the {namespace}tag notation in ElementPath expressions. However, advanced features like value comparison and functions are not available.
  """
    In addition to a full XPath implementation, lxml.etree supports the ElementPath language in the same way ElementTree does, even using (almost) the same implementation. The API provides four methods here that you can find on Elements and ElementTrees:
        - iterfind() iterates over all Elements that match the path expression
        - findall() returns a list of matching Elements
        - find() efficiently returns only the first match
        - findtext() returns the .text content of the first match
    As long as the tree is not modified, this path expression represents an identifier for a given element that can be used to find() it in the same tree later. 
  """
  # * Compared to XPath, ElementPath expressions have the advantage of being self-contained even for documents that use namespaces.
  root = etree.XML("<root><a x='123'>aText<b/><c/><b/></a></root>")
  tree = etree.ElementTree(root)
  if isDebuggingTutorial6[1]:
    print(root.find("b"), root.find("a").tag, sep="  :  ")
    print(root.find(".//b").tag, [b.tag for b in root.iterfind(".//b")],
          sep="  :  ")
    print(root.findall(".//a[@x]")[0].tag,
          root.findall(".//a[@y]"),
          sep="  :  ")
  """In lxml 3.4, there is a new helper to generate a structural ElementPath expression for an Element:
      """
    print(
        tree.getelementpath(root[0]),
        tree.getelementpath(root[0][0]),
        tree.getelementpath(root[0][1]),
        tree.getelementpath(root[0][2]),
    )
    print(tree.find(tree.getelementpath(root[0][2])) == root[0][2])
    print(etree.tostring(root, pretty_print=True).decode("unicode_escape"))
  """The .iter() method is a special case that only finds specific tags in the tree by their name, not based on a path. That means that the following commands are equivalent in the success case.
      """
    # *Note that the .find() method simply returns None if no match is found, whereas the other two examples would raise a StopIteration exception.
    print(
        root.find(".//b").tag,
        next(root.iterfind(".//b")).tag,
        next(root.iter("b")).tag,
    )

  ### print example about 1 ~ 2 topic
  """
        contentsY = contentsX.encode("raw_unicode_escape")
  """
  print("\n  ========== structure ==========\n")
  # contents = etree.tostring(root, pretty_print=True, encoding='UTF-8')
  etree.indent(root, space="  ")
  contents = etree.tostring(root, pretty_print=True)
  contentTree = etree.tostring(tree, pretty_print=True, xml_declaration=True)
  # * argument) xml_declaration=True; ValueError: Serialization to unicode must not request an XML declaration
  contentsX = contents.decode("unicode_escape")
  contentsY = contentTree.decode("unicode_escape")
  # # same # contentsX = codecs.decode(contents, "unicode_escape")
  # print(contents)

  # print(contentsX)
  # print(contentsY)


tutorial()
