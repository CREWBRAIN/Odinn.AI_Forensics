Usage
Support model
Here are some examples of using the support model API.

Import the library

import pypdfium2 as pdfium
Open a PDF using the helper class PdfDocument (supports file path strings, bytes, and byte buffers)

pdf = pdfium.PdfDocument("./path/to/document.pdf")
version = pdf.get_version()  # get the PDF standard version
n_pages = len(pdf)  # get the number of pages in the document
page = pdf[0]  # load a page
Render the page

bitmap = page.render(
    scale = 1,    # 72dpi resolution
    rotation = 0, # no additional rotation
    # ... further rendering options
)
pil_image = bitmap.to_pil()
pil_image.show()
Try some page methods

# Get page dimensions in PDF canvas units (1pt->1/72in by default)
width, height = page.get_size()
# Set the absolute page rotation to 90° clockwise
page.set_rotation(90)

# Locate objects on the page
for obj in page.get_objects():
    print(obj.level, obj.type, obj.get_pos())
Extract and search text

# Load a text page helper
textpage = page.get_textpage()

# Extract text from the whole page
text_all = textpage.get_text_range()
# Extract text from a specific rectangular area
text_part = textpage.get_text_bounded(left=50, bottom=100, right=width-50, top=height-100)

# Locate text on the page
searcher = textpage.search("something", match_case=False, match_whole_word=False)
# This returns the next occurrence as (char_index, char_count), or None if not found
first_occurrence = searcher.get_next()
Read the table of contents

for item in pdf.get_toc():
    state = "*" if item.n_kids == 0 else "-" if item.is_closed else "+"
    target = "?" if item.page_index is None else item.page_index+1
    print(
        "    " * item.level +
        "[%s] %s -> %s  # %s %s" % (
            state, item.title, target, item.view_mode, item.view_pos,
        )
    )
Create a new PDF with an empty A4 sized page

pdf = pdfium.PdfDocument.new()
width, height = (595, 842)
page_a = pdf.new_page(width, height)
Include a JPEG image in a PDF

pdf = pdfium.PdfDocument.new()

image = pdfium.PdfImage.new(pdf)
image.load_jpeg("./tests/resources/mona_lisa.jpg")
width, height = image.get_size()

matrix = pdfium.PdfMatrix().scale(width, height)
image.set_matrix(matrix)

page = pdf.new_page(width, height)
page.insert_obj(image)
page.gen_content()
Save the document

# PDF 1.7 standard
pdf.save("output.pdf", version=17)
Raw PDFium API
While helper classes conveniently wrap the raw PDFium API, it may still be accessed directly and is available in the namespace pypdfium2.raw. Lower-level helpers that may aid with using the raw API are provided in pypdfium2.internal.

import pypdfium2.raw as pdfium_c
import pypdfium2.internal as pdfium_i
Since PDFium is a large library, many components are not covered by helpers yet. You may seamlessly interact with the raw API while still using helpers where available. When used as ctypes function parameter, helper objects automatically resolve to the underlying raw object (but you may still access it explicitly if desired):

permission_flags = pdfium_c.FPDF_GetDocPermission(pdf.raw)  # explicit
permission_flags = pdfium_c.FPDF_GetDocPermission(pdf)      # implicit
For PDFium docs, please look at the comments in its public header files.3 A large variety of examples on how to interface with the raw API using ctypes is already provided with support model source code. Nonetheless, the following guide may be helpful to get started with the raw API, especially for developers who are not familiar with ctypes yet.

In general, PDFium functions can be called just like normal Python functions. However, parameters may only be passed positionally, i. e. it is not possible to use keyword arguments. There are no defaults, so you always need to provide a value for each argument.
# arguments: filepath (bytes), password (bytes|None)
# null-terminate filepath and encode as UTF-8
pdf = pdfium_c.FPDF_LoadDocument((filepath+"\x00").encode("utf-8"), None)
This is the underlying bindings declaration,4 which loads the function from the binary and contains the information required to convert Python types to their C equivalents.
if _libs["pdfium"].has("FPDF_LoadDocument", "cdecl"):
    FPDF_LoadDocument = _libs["pdfium"].get("FPDF_LoadDocument", "cdecl")
    FPDF_LoadDocument.argtypes = [FPDF_STRING, FPDF_BYTESTRING]
    FPDF_LoadDocument.restype = FPDF_DOCUMENT
Python bytes are converted to FPDF_STRING by ctypes autoconversion. When passing a string to a C function, it must always be null-terminated, as the function merely receives a pointer to the first item and then continues to read memory until it finds a null terminator.
While some functions are quite easy to use, things soon get more complex. First of all, function parameters are not only used for input, but also for output:

# Initialise an integer object (defaults to 0)
c_version = ctypes.c_int()
# Let the function assign a value to the c_int object, and capture its return code (True for success, False for failure)
ok = pdfium_c.FPDF_GetFileVersion(pdf, c_version)
# If successful, get the Python int by accessing the `value` attribute of the c_int object
# Otherwise, set the variable to None (in other cases, it may be desired to raise an exception instead)
version = c_version.value if ok else None
If an array is required as output parameter, you can initialise one like this (in general terms):

# long form
array_type = (c_type * array_length)
array_object = array_type()
# short form
array_object = (c_type * array_length)()
Example: Getting view mode and target position from a destination object returned by some other function.

# (Assuming `dest` is an FPDF_DEST)
n_params = ctypes.c_ulong()
# Create a C array to store up to four coordinates
view_pos = (pdfium_c.FS_FLOAT * 4)()
view_mode = pdfium_c.FPDFDest_GetView(dest, n_params, view_pos)
# Convert the C array to a Python list and cut it down to the actual number of coordinates
view_pos = list(view_pos)[:n_params.value]
For string output parameters, callers needs to provide a sufficiently long, pre-allocated buffer. This may work differently depending on what type the function requires, which encoding is used, whether the number of bytes or characters is returned, and whether space for a null terminator is included or not. Carefully review the documentation for the function in question to fulfill its requirements.

Example A: Getting the title string of a bookmark.

# (Assuming `bookmark` is an FPDF_BOOKMARK)
# First call to get the required number of bytes (not characters!), including space for a null terminator
n_bytes = pdfium_c.FPDFBookmark_GetTitle(bookmark, None, 0)
# Initialise the output buffer
buffer = ctypes.create_string_buffer(n_bytes)
# Second call with the actual buffer
pdfium_c.FPDFBookmark_GetTitle(bookmark, buffer, n_bytes)
# Decode to string, cutting off the null terminator
# Encoding: UTF-16LE (2 bytes per character)
title = buffer.raw[:n_bytes-2].decode("utf-16-le")
Example B: Extracting text in given boundaries.

# (Assuming `textpage` is an FPDF_TEXTPAGE and the boundary variables are set)
# Store common arguments for the two calls
args = (textpage, left, top, right, bottom)
# First call to get the required number of characters (not bytes!) - a possible null terminator is not included
n_chars = pdfium_c.FPDFText_GetBoundedText(*args, None, 0)
# If no characters were found, return an empty string
if n_chars <= 0:
    return ""
# Calculate the required number of bytes (UTF-16LE encoding again)
n_bytes = 2 * n_chars
# Initialise the output buffer - this function can work without null terminator, so skip it
buffer = ctypes.create_string_buffer(n_bytes)
# Re-interpret the type from char to unsigned short as required by the function
buffer_ptr = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ushort))
# Second call with the actual buffer
pdfium_c.FPDFText_GetBoundedText(*args, buffer_ptr, n_chars)
# Decode to string (You may want to pass `errors="ignore"` to skip possible errors in the PDF's encoding)
text = buffer.raw.decode("utf-16-le")
Not only are there different ways of string output that need to be handled according to the requirements of the function in question. String input, too, can work differently depending on encoding and type. We have already discussed FPDF_LoadDocument(), which takes a UTF-8 encoded string as char *. A different examples is FPDFText_FindStart(), which needs a UTF-16LE encoded string, given as unsigned short *:

# (Assuming `text` is a str and `textpage` an FPDF_TEXTPAGE)
# Add the null terminator and encode as UTF-16LE
enc_text = (text + "\x00").encode("utf-16-le")
# cast `enc_text` to a c_ushort pointer
text_ptr = ctypes.cast(enc_text, ctypes.POINTER(ctypes.c_ushort))
search = pdfium_c.FPDFText_FindStart(textpage, text_ptr, 0, 0)
Leaving strings, let's suppose you have a C memory buffer allocated by PDFium and wish to read its data. PDFium will provide you with a pointer to the first item of the byte array. To access the data, you'll want to re-interpret the pointer using ctypes.cast() to encompass the whole array:

# (Assuming `bitmap` is an FPDF_BITMAP and `size` is the expected number of bytes in the buffer)
buffer_ptr = pdfium_c.FPDFBitmap_GetBuffer(bitmap)
buffer_ptr = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_ubyte * size))
# Buffer as ctypes array (referencing the original buffer, will be unavailable as soon as the bitmap is destroyed)
c_array = buffer_ptr.contents
# Buffer as Python bytes (independent copy)
data = bytes(c_array)
Writing data from Python into a C buffer works in a similar fashion:

# (Assuming `buffer_ptr` is a pointer to the first item of a C buffer to write into,
#  `size` the number of bytes it can store, and `py_buffer` a Python byte buffer)
buffer_ptr = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_char * size))
# Read from the Python buffer, starting at its current position, directly into the C buffer
# (until the target is full or the end of the source is reached)
n_bytes = py_buffer.readinto(buffer_ptr.contents)  # returns the number of bytes read
If you wish to check whether two objects returned by PDFium are the same, the is operator won't help because ctypes does not have original object return (OOR), i. e. new, equivalent Python objects are created each time, although they might represent one and the same C object.5 That's why you'll want to use ctypes.addressof() to get the memory addresses of the underlying C object. For instance, this is used to avoid infinite loops on circular bookmark references when iterating through the document outline:

# (Assuming `pdf` is an FPDF_DOCUMENT)
seen = set()
bookmark = pdfium_c.FPDFBookmark_GetFirstChild(pdf, None)
while bookmark:
    # bookmark is a pointer, so we need to use its `contents` attribute to get the object the pointer refers to
    # (otherwise we'd only get the memory address of the pointer itself, which would result in random behaviour)
    address = ctypes.addressof(bookmark.contents)
    if address in seen:
        break  # circular reference detected
    else:
        seen.add(address)
    bookmark = pdfium_c.FPDFBookmark_GetNextSibling(pdf, bookmark)
In many situations, callback functions come in handy.6 Thanks to ctypes, it is seamlessly possible to use callbacks across Python/C language boundaries.

Example: Loading a document from a Python buffer. This way, file access can be controlled in Python while the whole data does not need to be in memory at once.

import os

# Factory class to create callable objects holding a reference to a Python buffer
class _reader_class:
  
  def __init__(self, py_buffer):
      self.py_buffer = py_buffer
  
  def __call__(self, _, position, p_buf, size):
      # Write data from Python buffer into C buffer, as explained before
      buffer_ptr = ctypes.cast(p_buf, ctypes.POINTER(ctypes.c_char * size))
      self.py_buffer.seek(position)
      self.py_buffer.readinto(buffer_ptr.contents)
      return 1  # non-zero return code for success

# (Assuming py_buffer is a Python file buffer, e. g. io.BufferedReader)
# Get the length of the buffer
py_buffer.seek(0, os.SEEK_END)
file_len = py_buffer.tell()
py_buffer.seek(0)

# Set up an interface structure for custom file access
fileaccess = pdfium_c.FPDF_FILEACCESS()
fileaccess.m_FileLen = file_len

# Assign the callback, wrapped in its CFUNCTYPE
fileaccess.m_GetBlock = type(fileaccess.m_GetBlock)( _reader_class(py_buffer) )

# Finally, load the document
pdf = pdfium_c.FPDF_LoadCustomDocument(fileaccess, None)
When using the raw API, special care needs to be taken regarding object lifetime, considering that Python may garbage collect objects as soon as their reference count reaches zero. However, the interpreter has no way of magically knowing how long the underlying resources of a Python object might still be needed on the C side, so measures need to be taken to keep such objects referenced until PDFium does not depend on them anymore.

If resources need to remain valid after the time of a function call, PDFium docs usually indicate this clearly. Ignoring requirements on object lifetime will lead to memory corruption (commonly resulting in a segfault).

For instance, the docs on FPDF_LoadCustomDocument() state that

The application must keep the file resources |pFileAccess| points to valid until the returned FPDF_DOCUMENT is closed. |pFileAccess| itself does not need to outlive the FPDF_DOCUMENT.

This means that the callback function and the Python buffer need to be kept alive as long as the FPDF_DOCUMENT is used. This can be achieved by referencing these objects in an accompanying class, e. g.

class PdfDataHolder:
    
    def __init__(self, buffer, function):
        self.buffer = buffer
        self.function = function
    
    def close(self):
        # Make sure both objects remain available until this function is called
        # No-op id() call to denote that the object needs to stay in memory up to this point
        id(self.function)
        self.buffer.close()

# ... set up an FPDF_FILEACCESS structure

# (Assuming `py_buffer` is the buffer and `fileaccess` the FPDF_FILEACCESS interface)
data_holder = PdfDataHolder(py_buffer, fileaccess.m_GetBlock)
pdf = pdfium_c.FPDF_LoadCustomDocument(fileaccess, None)

# ... work with the pdf

# Close the PDF to free resources
pdfium_c.FPDF_CloseDocument(pdf)
# Close the data holder, to keep the object itself and thereby the objects it
# references alive up to this point, as well as to release the buffer
data_holder.close()
Finally, let's finish with an example how to render the first page of a document to a PIL image in RGBA color format.

import math
import ctypes
import os.path
import PIL.Image
import pypdfium2.raw as pdfium_c

# Load the document
filepath = os.path.abspath("tests/resources/render.pdf")
pdf = pdfium_c.FPDF_LoadDocument((filepath+"\x00").encode("utf-8"), None)

# Check page count to make sure it was loaded correctly
page_count = pdfium_c.FPDF_GetPageCount(pdf)
assert page_count >= 1

# Load the first page and get its dimensions
page = pdfium_c.FPDF_LoadPage(pdf, 0)
width  = math.ceil(pdfium_c.FPDF_GetPageWidthF(page))
height = math.ceil(pdfium_c.FPDF_GetPageHeightF(page))

# Create a bitmap
# (Note, pdfium is faster at rendering transparency if we use BGRA rather than BGRx)
use_alpha = pdfium_c.FPDFPage_HasTransparency(page)
bitmap = pdfium_c.FPDFBitmap_Create(width, height, int(use_alpha))
# Fill the whole bitmap with a white background
# The color is given as a 32-bit integer in ARGB format (8 bits per channel)
pdfium_c.FPDFBitmap_FillRect(bitmap, 0, 0, width, height, 0xFFFFFFFF)

# Store common rendering arguments
render_args = (
    bitmap,  # the bitmap
    page,    # the page
    # positions and sizes are to be given in pixels and may exceed the bitmap
    0,       # left start position
    0,       # top start position
    width,   # horizontal size
    height,  # vertical size
    0,       # rotation (as constant, not in degrees!)
    pdfium_c.FPDF_LCD_TEXT | pdfium_c.FPDF_ANNOT,  # rendering flags, combined with binary or
)

# Render the page
pdfium_c.FPDF_RenderPageBitmap(*render_args)

# Get a pointer to the first item of the buffer
buffer_ptr = pdfium_c.FPDFBitmap_GetBuffer(bitmap)
# Re-interpret the pointer to encompass the whole buffer
buffer_ptr = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_ubyte * (width * height * 4)))

# Create a PIL image from the buffer contents
img = PIL.Image.frombuffer("RGBA", (width, height), buffer_ptr.contents, "raw", "BGRA", 0, 1)
# Save it as file
img.save("out.png")

# Free resources
pdfium_c.FPDFBitmap_Destroy(bitmap)
pdfium_c.FPDF_ClosePage(page)
pdfium_c.FPDF_CloseDocument(pdf)
Command-line Interface
pypdfium2 also ships with a simple command-line interface, providing access to key features of the support model in a shell environment (e. g. rendering, content extraction, document inspection, page rearranging, ...).

The primary motivation behind this is to have a nice testing interface, but it may be helpful in a variety of other situations as well. Usage should be largely self-explanatory, assuming a minimum of familiarity with the command-line.