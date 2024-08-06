# TESSERACT(1) Manual Page

## NAME
**tesseract** - command-line OCR engine

## SYNOPSIS
`tesseract FILE OUTPUTBASE [OPTIONS]… [CONFIGFILE]…`

## DESCRIPTION
`tesseract(1)` is a commercial quality OCR engine originally developed at HP between 1985 and 1995. In 1995, this engine was among the top 3 evaluated by UNLV. It was open-sourced by HP and UNLV in 2005, and has been developed at Google since then.

## IN/OUT ARGUMENTS

### FILE
The name of the input file. This can either be an image file or a text file. Most image file formats (anything readable by Leptonica) are supported. A text file lists the names of all input images (one image name per line). The results will be combined in a single file for each output file format (txt, pdf, hocr, xml). If FILE is `stdin` or `-` then the standard input is used.

### OUTPUTBASE
The basename of the output file (to which the appropriate extension will be appended). By default, the output will be a text file with `.txt` added to the basename unless there are one or more parameters set which explicitly specify the desired output. If OUTPUTBASE is `stdout` or `-` then the standard output is used.

## OPTIONS

### -c CONFIGVAR=VALUE
Set value for parameter CONFIGVAR to VALUE. Multiple `-c` arguments are allowed.

### --dpi N
Specify the resolution N in DPI for the input image(s). A typical value for N is 300. Without this option, the resolution is read from the metadata included in the image. If an image does not include that information, Tesseract tries to guess it.

### -l LANG
### -l SCRIPT
The language or script to use. If none is specified, `eng` (English) is assumed. Multiple languages may be specified, separated by plus characters. Tesseract uses 3-character ISO 639-2 language codes.

### --psm N
Set Tesseract to only run a subset of layout analysis and assume a certain form of image. The options for N are:

- `0` = Orientation and script detection (OSD) only.
- `1` = Automatic page segmentation with OSD.
- `2` = Automatic page segmentation, but no OSD, or OCR. (not implemented)
- `3` = Fully automatic page segmentation, but no OSD. (Default)
- `4` = Assume a single column of text of variable sizes.
- `5` = Assume a single uniform block of vertically aligned text.
- `6` = Assume a single uniform block of text.
- `7` = Treat the image as a single text line.
- `8` = Treat the image as a single word.
- `9` = Treat the image as a single word in a circle.
- `10` = Treat the image as a single character.
- `11` = Sparse text. Find as much text as possible in no particular order.
- `12` = Sparse text with OSD.
- `13` = Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

### --oem N
Specify OCR Engine mode. The options for N are:

- `0` = Original Tesseract only.
- `1` = Neural nets LSTM only.
- `2` = Tesseract + LSTM.
- `3` = Default, based on what is available.

### --tessdata-dir PATH
Specify the location of tessdata path.

### --user-patterns FILE
Specify the location of user patterns file.

### --user-words FILE
Specify the location of user words file.

## CONFIGFILE
The name of a config to use. The name can be a file in `tessdata/configs` or `tessdata/tessconfigs`, or an absolute or relative file path. A config is a plain text file which contains a list of parameters and their values, one per line, with a space separating parameter from value. Interesting config files include:

- `alto` — Output in ALTO format (`OUTPUTBASE.xml`).
- `hocr` — Output in hOCR format (`OUTPUTBASE.hocr`).
- `pdf` — Output PDF (`OUTPUTBASE.pdf`).
- `tsv` — Output TSV (`OUTPUTBASE.tsv`).
- `txt` — Output plain text (`OUTPUTBASE.txt`).
- `get.images` — Write processed input images to file (`OUTPUTBASE.processedPAGENUMBER.tif`).
- `logfile` — Redirect debug messages to file (`tesseract.log`).
- `lstm.train` — Output files used by LSTM training (`OUTPUTBASE.lstmf`).
- `makebox` — Write box file (`OUTPUTBASE.box`).
- `quiet` — Redirect debug messages to `/dev/null`.

It is possible to select several config files, for example `tesseract image.png demo alto hocr pdf txt` will create four output files `demo.alto`, `demo.hocr`, `demo.pdf` and `demo.txt` with the OCR results.

**Nota bene:** The options `-l LANG`, `-l SCRIPT` and `--psm N` must occur before any CONFIGFILE.

## SINGLE OPTIONS

### -h, --help
Show help message.

### --help-extra
Show extra help for advanced users.

### --help-psm
Show page segmentation modes.

### --help-oem
Show OCR Engine modes.

### -v, --version
Returns the current version of the `tesseract(1)` executable.

### --list-langs
List available languages for tesseract engine. Can be used with `--tessdata-dir PATH`.

### --print-parameters
Print tesseract parameters.

## LANGUAGES AND SCRIPTS
To recognize some text with Tesseract, it is normally necessary to specify the language(s) or script(s) of the text (unless it is English text which is supported by default) using `-l LANG` or `-l SCRIPT`.

Selecting a language automatically also selects the language-specific character set and dictionary (word list).

Selecting a script typically selects all characters of that script which can be from different languages. The dictionary which is included also contains a mix from different languages. In most cases, a script also supports English. So it is possible to recognize a language that has not been specifically trained for by using `traineddata` for the script it is written in.

More than one language or script may be specified by using `+`. Example: `tesseract myimage.png myimage -l eng+deu+fra`.

To use a non-standard language pack named `foo.traineddata`, set the `TESSDATA_PREFIX` environment variable so the file can be found at `TESSDATA_PREFIX/tessdata/foo.traineddata` and give Tesseract the argument `-l foo`.

## CONFIG FILES AND AUGMENTING WITH USER DATA
Tesseract config files consist of lines with parameter-value pairs (space separated). The parameters are documented as flags in the source code like the following one in `tesseractclass.h`:
