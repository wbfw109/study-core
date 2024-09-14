# %%
import subprocess
import tempfile
from typing import Optional, Union

import requests
from bs4 import BeautifulSoup, ResultSet, Tag
from IPython.core.interactiveshell import InteractiveShell

# Enable all outputs in the Jupyter notebook environment
InteractiveShell.ast_node_interactivity = "all"


# List of base URLs for the websites to fetch content from
base_urls = [
    "https://en.cppreference.com/w/c",
    "https://en.cppreference.com/w/cpp",
    # Add more URLs here as needed
]


# Function to fetch the HTML content from a given URL
def fetch_html(url: str) -> str:
    """Fetches the HTML content of the given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch page: {response.status_code}")


# Function to parse and format the output of 'rowtop' class rows
def parse_row_top(
    element: Union[Tag, None], base_url: str, indent_level: int = 0
) -> str:
    """Parse the 'rowtop' class tr tag content."""
    result = ""
    p_tag = element.find("p") if element else None
    if isinstance(p_tag, Tag):
        text = (
            p_tag.get_text(separator=" ", strip=True)
            .replace("\u00a0", " ")
            .replace(" , ", ", ")
        )

        parts = text.split("│")
        header = parts[0].strip()
        footer = parts[1].strip() if len(parts) > 1 else ""

        result += f"{' ' * indent_level}# {header}  │  {footer}\n"

        a_tags = p_tag.find_all("a")
        for a_tag in a_tags:
            if isinstance(a_tag, Tag):
                link = a_tag.get("href")
                link_text = a_tag.get_text(strip=True)
                result += f"{' ' * indent_level}  ⚓ {link_text} ; {base_url}{link}\n"
    return result


# Function to extract and format links in td tags
def extract_links(
    element: Union[Tag, None], base_url: str, indent_level: int = 0
) -> str:
    """Extract and format links from td tag."""
    result = ""
    if not element:
        return result

    for sibling in element.children:
        if isinstance(sibling, Tag) and sibling.name == "p":
            a_tag = sibling.find("a")
            if isinstance(a_tag, Tag):
                link = a_tag.get("href")
                text = a_tag.get_text(strip=True)
                result += f"{' ' * indent_level}⚓ {text} ; {base_url}{link}\n"

        if (
            isinstance(sibling, Tag)
            and sibling.name == "div"
            and "mainpagediv" in sibling.get("class", [])
        ):
            for sub_link in sibling.find_all("a"):
                if isinstance(sub_link, Tag):
                    sub_text = sub_link.get_text(strip=True)
                    sub_href = sub_link.get("href")
                    result += f"{' ' * (indent_level + 2)}⚓ {sub_text} ; {base_url}{sub_href}\n"
    return result


# Function to extract 'Technical specifications'
def extract_technical_specifications(
    element: Union[Tag, None], base_url: str, indent_level: int = 0
) -> str:
    """Parse the 'Technical specifications' section."""
    result = ""
    if element:
        a_tag = element.find("a", title="c/experimental")
        if isinstance(a_tag, Tag) and "Technical specifications" in a_tag.get_text(
            strip=True
        ):
            result += f"{' ' * indent_level}⚓ {a_tag.get_text(strip=True)} ; {base_url}{a_tag.get('href')}\n"

            for sub_tag in element.find_all("p"):
                sub_links = sub_tag.find_all("a")
                for sub_link_item in sub_links:
                    if isinstance(sub_link_item, Tag):
                        sub_text = sub_link_item.get_text(strip=True)
                        sub_href = sub_link_item.get("href")
                        result += f"{' ' * (indent_level + 2)}⚓ {sub_text} ; {base_url}{sub_href}\n"
    return result


# Function to extract and format external links
def extract_external_links(
    element: Union[Tag, None], base_url: str, indent_level: int = 0
) -> str:
    """Extract and format external links."""
    result = ""
    if element:
        a_tags = element.find_all("a")
        for a_tag in a_tags:
            if isinstance(a_tag, Tag):
                link = a_tag.get("href")
                text = a_tag.get_text(strip=True)
                result += f"{' ' * indent_level}⚓ {text} ; {base_url}{link}\n"
    return result


# Function to process the HTML and extract relevant sections
def process_html(soup: BeautifulSoup, base_url: str) -> str:
    """Processes the parsed HTML to extract various sections and links."""
    result = ""

    title_tag: Optional[Tag] = soup.find("h1", {"id": "firstHeading"})
    if title_tag:
        result += f"⚓ {title_tag.get_text(strip=True)} ; {base_url}/w/c\n"

    tr_tags: ResultSet[Tag] = soup.find_all("tr")
    for tr_tag in tr_tags:
        if "rowtop" in tr_tag.get("class", []):
            result += parse_row_top(tr_tag, base_url, indent_level=2)
        else:
            td_tags: ResultSet[Tag] = tr_tag.find_all("td")
            for td_tag in td_tags:
                if td_tag.find("a", title="c/experimental") and not td_tag.find(
                    "a", string="Technical Specifications"
                ):
                    result += extract_technical_specifications(
                        td_tag, base_url, indent_level=2
                    )
                else:
                    result += extract_links(td_tag, base_url, indent_level=2)

        if "rowbottom" in tr_tag.get("class", []):
            td_tags = tr_tag.find_all("td")
            for td_tag in td_tags:
                result += extract_external_links(td_tag, base_url, indent_level=2)
    return result


# Function to save content to a temp file and open it in VSCode
def save_to_temp_and_open(content: str) -> None:
    """Save parsed content to a temporary file and open it with VSCode."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", dir="/tmp"
    ) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(content.encode())

    # Open the file with VSCode
    subprocess.run(["code", temp_file_path])


# Main function to parse multiple websites
def parse_multiple_sites(urls: list) -> None:
    """Fetches and parses multiple URLs, saves the output to a file, and opens it."""
    for url in urls:
        html_content: str = fetch_html(url)
        soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")
        parsed_content: str = process_html(soup, url)
        save_to_temp_and_open(parsed_content)


# Run the parsing function for multiple websites
parse_multiple_sites(base_urls)


# %%
html_content = """
<div id="content">
                <a id="top"></a>
                <div id="mw-js-message" style="display:none;"></div>
                                <!-- firstHeading -->
<script async="" type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CKYITK7M&amp;placement=cppreferencecom" id="_carbonads_js"></script>

<script async="" src="https://www.googletagmanager.com/gtag/js?id=G-8HW0LXMYCY"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-8HW0LXMYCY');
</script>
                <h1 id="firstHeading" class="firstHeading">C reference</h1>
                <!-- /firstHeading -->
                <!-- bodyContent -->
                <div id="bodyContent">
                                        <!-- tagline -->
                    <div id="siteSub">From cppreference.com</div>
                    <!-- /tagline -->
                                        <!-- subtitle -->
                    <div id="contentSub"></div>
                    <!-- /subtitle -->
                                                            <!-- bodycontent -->
                    <div id="mw-content-text" lang="en" dir="ltr" class="mw-content-ltr"><div class="t-navbar" style=""><div class="t-navbar-sep">&nbsp;</div><div class="t-navbar-head"><strong class="selflink"> C</strong><div class="t-navbar-menu"><div><div><table class="t-nv-begin" cellpadding="0" style="line-height:1.1em;">
<tbody><tr class="t-nv"><td colspan="5"><a href="/w/c/compiler_support" title="c/compiler support">Compiler support</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/language" title="c/language">Language</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/header" title="c/header">Headers</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/types" title="c/types">Type support</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/program" title="c/program">Program utilities</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/variadic" title="c/variadic">Variadic function support</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/error" title="c/error">Error handling</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/memory" title="c/memory">Dynamic memory management</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/string" title="c/string">Strings library</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/algorithm" title="c/algorithm">Algorithms</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/numeric" title="c/numeric">Numerics</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/chrono" title="c/chrono">Date and time utilities</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/io" title="c/io">Input/output support</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/locale" title="c/locale">Localization support</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/thread" title="c/thread">Concurrency support</a> <span class="t-mark-rev t-since-c11">(C11)</span></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/experimental" title="c/experimental">Technical Specifications</a></td></tr>
<tr class="t-nv"><td colspan="5"><a href="/w/c/symbol_index" title="c/symbol index">Symbol index</a></td></tr>
</tbody></table></div><div><span class="editsection noprint plainlinks" title="Edit this template"><a rel="nofollow" class="external text" href="https://en.cppreference.com/mwiki/index.php?title=Template:c/navbar_content&amp;action=edit">[edit]</a></span></div></div></div></div><div class="t-navbar-sep">&nbsp;</div></div>
<table class="mainpagetable" cellspacing="0" style="width:100%; white-space:nowrap;">

<tbody><tr class="row rowtop">
<td colspan="3">
<p><span style="font-size: 0.8em;">C89, <a href="/w/c/95" title="c/95">C95</a>, <a href="/w/c/99" title="c/99">C99</a>, <a href="/w/c/11" title="c/11">C11</a>, <a href="/w/c/17" title="c/17">C17</a>, <a href="/w/c/23" title="c/23">C23</a>&nbsp;&nbsp;│&nbsp;&nbsp;<a href="/w/c/compiler_support" title="c/compiler support">Compiler support</a> <a href="/w/c/compiler_support/99" title="c/compiler support/99">C99</a>, <a href="/w/c/compiler_support/23" title="c/compiler support/23">C23</a></span>
</p>
</td></tr>
<tr class="row">
<td>
<p><b><a href="/w/c/language" title="c/language"> Language</a></b>
</p>
<div class="mainpagediv">
<p><a href="/w/c/language/basic_concepts" title="c/language/basic concepts"> Basic concepts</a><br>
<a href="/w/c/keyword" title="c/keyword"> Keywords</a><br>
<a href="/w/c/preprocessor" title="c/preprocessor"> Preprocessor</a><br>
<a href="/w/c/language/expressions" title="c/language/expressions"> Expressions</a><br>
<a href="/w/c/language/declarations" title="c/language/declarations"> Declaration</a><br>
<a href="/w/c/language/initialization" title="c/language/initialization"> Initialization</a><br>
<a href="/w/c/language/functions" title="c/language/functions"> Functions</a><br>
<a href="/w/c/language/statements" title="c/language/statements"> Statements</a><br>
</p>
</div>
<p><b><a href="/w/c/header" title="c/header"> Headers</a></b>
</p>
</td>
<td>
<p><b><a href="/w/c/types" title="c/types"> Type support</a></b>
</p><p><b><a href="/w/c/program" title="c/program"> Program utilities</a></b>
</p><p><b><a href="/w/c/variadic" title="c/variadic"> Variadic functions</a></b>
</p><p><b><a href="/w/c/error" title="c/error"> Diagnostics library</a></b>
</p><p><b><a href="/w/c/memory" title="c/memory"> Dynamic memory management</a></b>
</p><p><b><a href="/w/c/string" title="c/string"> Strings library</a></b>
</p>
<div class="mainpagediv">
<p>Null-terminated strings:<br>
&nbsp;&nbsp; <a href="/w/c/string/byte" title="c/string/byte">byte</a>&nbsp;&nbsp;−&nbsp;&nbsp; <a href="/w/c/string/multibyte" title="c/string/multibyte">multibyte</a>&nbsp;&nbsp;−&nbsp;&nbsp; <a href="/w/c/string/wide" title="c/string/wide">wide</a>
</p>
</div>
<p><b><a href="/w/c/algorithm" title="c/algorithm"> Algorithms library</a></b>
</p>
</td>
<td>
<p><b><a href="/w/c/numeric" title="c/numeric"> Numerics library</a></b>
</p>
<div class="mainpagediv">
<p><a href="/w/c/numeric/math" title="c/numeric/math"> Common mathematical functions</a><br>
<a href="/w/c/numeric/fenv" title="c/numeric/fenv"> Floating-point environment</a> <span class="t-mark-rev t-since-c99">(C99)</span><br>
<a href="/w/c/numeric/random" title="c/numeric/random"> Pseudo-random number generation</a><br>
<a href="/w/c/numeric/complex" title="c/numeric/complex"> Complex number arithmetic</a> <span class="t-mark-rev t-since-c99">(C99)</span><br>
<a href="/w/c/numeric/tgmath" title="c/numeric/tgmath"> Type-generic math</a> <span class="t-mark-rev t-since-c99">(C99)</span>
</p>
</div>
<p><b><a href="/w/c/chrono" title="c/chrono"> Date and time library</a></b>
</p><p><b><a href="/w/c/locale" title="c/locale"> Localization library</a></b>
</p><p><b><a href="/w/c/io" title="c/io"> Input/output library</a></b>
</p><p><b><a href="/w/c/thread" title="c/thread"> Concurrency support library</a></b> <span class="t-mark-rev t-since-c11">(C11)</span>
</p>
</td></tr>
<tr class="row">
<td colspan="3"><b><a href="/w/c/experimental" title="c/experimental"> Technical specifications</a></b>
<p>&nbsp;&nbsp; <b><a href="/w/c/experimental/dynamic" title="c/experimental/dynamic">Dynamic memory extensions</a></b>&nbsp;&nbsp;<span class="t-mark-rev t-since-dynamic-tr t-mark-ts">(dynamic memory TR)</span><br>
&nbsp;&nbsp; <b><a href="/w/c/experimental/fpext1" title="c/experimental/fpext1">Floating-point extensions, Part 1</a></b>&nbsp;&nbsp;<span class="t-mark-rev t-since-fpext1-ts t-mark-ts">(FP Ext 1 TS)</span><br>
&nbsp;&nbsp; <b><a href="/w/c/experimental/fpext4" title="c/experimental/fpext4">Floating-point extensions, Part 4</a></b>&nbsp;&nbsp;<span class="t-mark-rev t-since-fpext4-ts t-mark-ts">(FP Ext 4 TS)</span><br>
</p>
</td></tr>
<tr class="row rowbottom">
<td colspan="3"> <a href="/w/c/links" title="c/links">External Links</a>&nbsp;&nbsp;−&nbsp;&nbsp;<a href="/w/c/links/libs" title="c/links/libs">Non-ANSI/ISO Libraries</a>&nbsp;&nbsp;−&nbsp;&nbsp;<a href="/w/c/index" title="c/index" class="mw-redirect">Index</a>&nbsp;&nbsp;−&nbsp;&nbsp;<a href="/w/c/symbol_index" title="c/symbol index">Symbol Index</a>
</td></tr></tbody></table>

<!-- 
NewPP limit report
Preprocessor visited node count: 402/1000000
Preprocessor generated node count: 1719/1000000
Post‐expand include size: 14288/4194304 bytes
Template argument size: 968/4194304 bytes
Highest expansion depth: 13/40
Expensive parser function count: 0/100
-->

<!-- Saved in parser cache with key mwiki1-mwiki_en_:pcache:idhash:5742-0!*!0!*!*!*!* and timestamp 20240801004832 -->
</div>                    <!-- /bodycontent -->
                                        <!-- printfooter -->
                    <div class="printfooter">
                    Retrieved from "<a href="https://en.cppreference.com/mwiki/index.php?title=c&amp;oldid=94077">https://en.cppreference.com/mwiki/index.php?title=c&amp;oldid=94077</a>"                    </div>
                    <!-- /printfooter -->
                                                            <!-- catlinks -->
                    <div id="catlinks" class="catlinks catlinks-allhidden"></div>                    <!-- /catlinks -->
                                                            <div class="visualClear"></div>
                    <!-- debughtml -->
                                        <!-- /debughtml -->
                </div>
                <!-- /bodyContent -->
            </div>
"""
