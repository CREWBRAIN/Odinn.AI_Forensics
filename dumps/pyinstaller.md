Using PyInstaller
The syntax of the pyinstaller command is:

pyinstaller [options] script [script …] | specfile

In the most simple case, set the current directory to the location of your program myscript.py and execute:

pyinstaller myscript.py
PyInstaller analyzes myscript.py and:

Writes myscript.spec in the same folder as the script.

Creates a folder build in the same folder as the script if it does not exist.

Writes some log files and working files in the build folder.

Creates a folder dist in the same folder as the script if it does not exist.

Writes the myscript executable folder in the dist folder.

In the dist folder you find the bundled app you distribute to your users.

Normally you name one script on the command line. If you name more, all are analyzed and included in the output. However, the first script named supplies the name for the spec file and for the executable folder or file. Its code is the first to execute at run-time.

For certain uses you may edit the contents of myscript.spec (described under Using Spec Files). After you do this, you name the spec file to PyInstaller instead of the script:

pyinstaller myscript.spec

The myscript.spec file contains most of the information provided by the options that were specified when pyinstaller (or pyi-makespec) was run with the script file as the argument. You typically do not need to specify any options when running pyinstaller with the spec file. Only a few command-line options have an effect when building from a spec file.

You may give a path to the script or spec file, for example

pyinstaller options… ~/myproject/source/myscript.py

or, on Windows,

pyinstaller "C:\Documents and Settings\project\myscript.spec"

Options
A full list of the pyinstaller command’s options are as follows:

Positional Arguments
scriptname
Name of scriptfiles to be processed or exactly one .spec file. If a .spec file is specified, most options are unnecessary and are ignored.

Options
-h, --help
show this help message and exit

-v, --version
Show program version info and exit.

--distpath DIR
Where to put the bundled app (default: ./dist)

--workpath WORKPATH
Where to put all the temporary work files, .log, .pyz and etc. (default: ./build)

-y, --noconfirm
Replace output directory (default: SPECPATH/dist/SPECNAME) without asking for confirmation

--upx-dir UPX_DIR
Path to UPX utility (default: search the execution path)

--clean
Clean PyInstaller cache and remove temporary files before building.

--log-level LEVEL
Amount of detail in build-time console messages. LEVEL may be one of TRACE, DEBUG, INFO, WARN, DEPRECATION, ERROR, FATAL (default: INFO). Also settable via and overrides the PYI_LOG_LEVEL environment variable.

What To Generate
-D, --onedir
Create a one-folder bundle containing an executable (default)

-F, --onefile
Create a one-file bundled executable.

--specpath DIR
Folder to store the generated spec file (default: current directory)

-n NAME, --name NAME
Name to assign to the bundled app and spec file (default: first script’s basename)

--contents-directory CONTENTS_DIRECTORY
For onedir builds only, specify the name of the directory in which all supporting files (i.e. everything except the executable itself) will be placed in. Use “.” to re-enable old onedir layout without contents directory.

What To Bundle, Where To Search
--add-data SOURCE:DEST
Additional data files or directories containing data files to be added to the application. The argument value should be in form of “source:dest_dir”, where source is the path to file (or directory) to be collected, dest_dir is the destination directory relative to the top-level application directory, and both paths are separated by a colon (:). To put a file in the top-level application directory, use . as a dest_dir. This option can be used multiple times.

--add-binary SOURCE:DEST
Additional binary files to be added to the executable. See the --add-data option for the format. This option can be used multiple times.

-p DIR, --paths DIR
A path to search for imports (like using PYTHONPATH). Multiple paths are allowed, separated by ':', or use this option multiple times. Equivalent to supplying the pathex argument in the spec file.

--hidden-import MODULENAME, --hiddenimport MODULENAME
Name an import not visible in the code of the script(s). This option can be used multiple times.

--collect-submodules MODULENAME
Collect all submodules from the specified package or module. This option can be used multiple times.

--collect-data MODULENAME, --collect-datas MODULENAME
Collect all data from the specified package or module. This option can be used multiple times.

--collect-binaries MODULENAME
Collect all binaries from the specified package or module. This option can be used multiple times.

--collect-all MODULENAME
Collect all submodules, data files, and binaries from the specified package or module. This option can be used multiple times.

--copy-metadata PACKAGENAME
Copy metadata for the specified package. This option can be used multiple times.

--recursive-copy-metadata PACKAGENAME
Copy metadata for the specified package and all its dependencies. This option can be used multiple times.

--additional-hooks-dir HOOKSPATH
An additional path to search for hooks. This option can be used multiple times.

--runtime-hook RUNTIME_HOOKS
Path to a custom runtime hook file. A runtime hook is code that is bundled with the executable and is executed before any other code or module to set up special features of the runtime environment. This option can be used multiple times.

--exclude-module EXCLUDES
Optional module or package (the Python name, not the path name) that will be ignored (as though it was not found). This option can be used multiple times.

--splash IMAGE_FILE
(EXPERIMENTAL) Add an splash screen with the image IMAGE_FILE to the application. The splash screen can display progress updates while unpacking.

How To Generate
-d {all,imports,bootloader,noarchive}, --debug {all,imports,bootloader,noarchive}
Provide assistance with debugging a frozen application. This argument may be provided multiple times to select several of the following options. - all: All three of the following options. - imports: specify the -v option to the underlying Python interpreter, causing it to print a message each time a module is initialized, showing the place (filename or built-in module) from which it is loaded. See https://docs.python.org/3/using/cmdline.html#id4. - bootloader: tell the bootloader to issue progress messages while initializing and starting the bundled app. Used to diagnose problems with missing imports. - noarchive: instead of storing all frozen Python source files as an archive inside the resulting executable, store them as files in the resulting output directory.

--optimize LEVEL
Bytecode optimization level used for collected python modules and scripts. For details, see the section “Bytecode Optimization Level” in PyInstaller manual.

--python-option PYTHON_OPTION
Specify a command-line option to pass to the Python interpreter at runtime. Currently supports “v” (equivalent to “–debug imports”), “u”, “W <warning control>”, “X <xoption>”, and “hash_seed=<value>”. For details, see the section “Specifying Python Interpreter Options” in PyInstaller manual.

-s, --strip
Apply a symbol-table strip to the executable and shared libs (not recommended for Windows)

--noupx
Do not use UPX even if it is available (works differently between Windows and *nix)

--upx-exclude FILE
Prevent a binary from being compressed when using upx. This is typically used if upx corrupts certain binaries during compression. FILE is the filename of the binary without path. This option can be used multiple times.

Windows And Mac Os X Specific Options
-c, --console, --nowindowed
Open a console window for standard i/o (default). On Windows this option has no effect if the first script is a ‘.pyw’ file.

-w, --windowed, --noconsole
Windows and Mac OS X: do not provide a console window for standard i/o. On Mac OS this also triggers building a Mac OS .app bundle. On Windows this option is automatically set if the first script is a ‘.pyw’ file. This option is ignored on *NIX systems.

--hide-console {hide-early,minimize-late,hide-late,minimize-early}
Windows only: in console-enabled executable, have bootloader automatically hide or minimize the console window if the program owns the console window (i.e., was not launched from an existing console window).

-i <FILE.ico or FILE.exe,ID or FILE.icns or Image or "NONE">, --icon <FILE.ico or FILE.exe,ID or FILE.icns or Image or "NONE">
FILE.ico: apply the icon to a Windows executable. FILE.exe,ID: extract the icon with ID from an exe. FILE.icns: apply the icon to the .app bundle on Mac OS. If an image file is entered that isn’t in the platform format (ico on Windows, icns on Mac), PyInstaller tries to use Pillow to translate the icon into the correct format (if Pillow is installed). Use “NONE” to not apply any icon, thereby making the OS show some default (default: apply PyInstaller’s icon). This option can be used multiple times.

--disable-windowed-traceback
Disable traceback dump of unhandled exception in windowed (noconsole) mode (Windows and macOS only), and instead display a message that this feature is disabled.

Windows Specific Options
--version-file FILE
Add a version resource from FILE to the exe.

-m <FILE or XML>, --manifest <FILE or XML>
Add manifest FILE or XML to the exe.

-r RESOURCE, --resource RESOURCE
Add or update a resource to a Windows executable. The RESOURCE is one to four items, FILE[,TYPE[,NAME[,LANGUAGE]]]. FILE can be a data file or an exe/dll. For data files, at least TYPE and NAME must be specified. LANGUAGE defaults to 0 or may be specified as wildcard * to update all resources of the given TYPE and NAME. For exe/dll files, all resources from FILE will be added/updated to the final executable if TYPE, NAME and LANGUAGE are omitted or specified as wildcard *. This option can be used multiple times.

--uac-admin
Using this option creates a Manifest that will request elevation upon application start.

--uac-uiaccess
Using this option allows an elevated application to work with Remote Desktop.

Mac Os Specific Options
--argv-emulation
Enable argv emulation for macOS app bundles. If enabled, the initial open document/URL event is processed by the bootloader and the passed file paths or URLs are appended to sys.argv.

--osx-bundle-identifier BUNDLE_IDENTIFIER
Mac OS .app bundle identifier is used as the default unique program name for code signing purposes. The usual form is a hierarchical name in reverse DNS notation. For example: com.mycompany.department.appname (default: first script’s basename)

--target-architecture ARCH, --target-arch ARCH
Target architecture (macOS only; valid values: x86_64, arm64, universal2). Enables switching between universal2 and single-arch version of frozen application (provided python installation supports the target architecture). If not target architecture is not specified, the current running architecture is targeted.

--codesign-identity IDENTITY
Code signing identity (macOS only). Use the provided identity to sign collected binaries and generated executable. If signing identity is not provided, ad- hoc signing is performed instead.

--osx-entitlements-file FILENAME
Entitlements file to use when code-signing the collected binaries (macOS only).

Rarely Used Special Options
--runtime-tmpdir PATH
Where to extract libraries and support files in onefile mode. If this option is given, the bootloader will ignore any temp-folder location defined by the run-time OS. The _MEIxxxxxx-folder will be created here. Please use this option only if you know what you are doing. Note that on POSIX systems, PyInstaller’s bootloader does NOT perform shell-style environment variable expansion on the given path string. Therefore, using environment variables (e.g., ~ or $HOME) in path will NOT work.

--bootloader-ignore-signals
Tell the bootloader to ignore signals rather than forwarding them to the child process. Useful in situations where for example a supervisor process signals both the bootloader and the child (e.g., via a process group) to avoid signalling the child twice.

Shortening the Command
Because of its numerous options, a full pyinstaller command can become very long. You will run the same command again and again as you develop your script. You can put the command in a shell script or batch file, using line continuations to make it readable. For example, in GNU/Linux:

pyinstaller --noconfirm --log-level=WARN \
    --onefile --nowindow \
    --add-data="README:." \
    --add-data="image1.png:img" \
    --add-binary="libfoo.so:lib" \
    --hidden-import=secret1 \
    --hidden-import=secret2 \
    --upx-dir=/usr/local/share/ \
    myscript.spec
Or in Windows, use the little-known BAT file line continuation:

pyinstaller --noconfirm --log-level=WARN ^
    --onefile --nowindow ^
    --add-data="README:." ^
    --add-data="image1.png:img" ^
    --add-binary="libfoo.so:lib" ^
    --hidden-import=secret1 ^
    --hidden-import=secret2 ^
    --icon=..\MLNMFLCN.ICO ^
    myscript.spec
Running PyInstaller from Python code
If you want to run PyInstaller from Python code, you can use the run function defined in PyInstaller.__main__. For instance, the following code:

import PyInstaller.__main__

PyInstaller.__main__.run([
    'my_script.py',
    '--onefile',
    '--windowed'
])
Is equivalent to:

pyinstaller my_script.py --onefile --windowed
Using UPX
UPX is a free utility for compressing executable files and libraries. It is available for most operating systems and can compress a large number of executable file formats. See the UPX home page for downloads, and for the list of supported file formats.

When UPX is available, PyInstaller uses it to individually compress each collected binary file (executable, shared library, or python extension) in order to reduce the overall size of the frozen application (the one-dir bundle directory, or the one-file executable). The frozen application’s executable itself is not UPX-compressed (regardless of one-dir or one-file mode), as most of its size comprises the embedded archive that already contains individually compressed files.

PyInstaller looks for the UPX in the standard executable path(s) (defined by PATH environment variable), or in the path specified via the --upx-dir command-line option. If found, it is used automatically. The use of UPX can be completely disabled using the --noupx command-line option.

Note

UPX is currently used only on Windows. On other operating systems, the collected binaries are not processed even if UPX is found. The shared libraries (e.g., the Python shared library) built on modern linux distributions seem to break when processed with UPX, resulting in defunct application bundles. On macOS, UPX currently fails to process .dylib shared libraries; furthermore the UPX-compressed files fail the validation check of the codesign utility, and therefore cannot be code-signed (which is a requirement on the Apple M1 platform).

Excluding problematic files from UPX processing
Using UPX may end up corrupting a collected shared library. Known examples of such corruption are Windows DLLs with Control Flow Guard (CFG) enabled, as well as Qt5 and Qt6 plugins. In such cases, individual files may be need to be excluded from UPX processing, using the --upx-exclude option (or using the upx_exclude argument in the .spec file).

Changed in version 4.2: PyInstaller detects CFG-enabled DLLs and automatically excludes them from UPX processing.

Changed in version 4.3: PyInstaller automatically excludes Qt5 and Qt6 plugins from UPX processing.

Although PyInstaller attempts to automatically detect and exclude some of the problematic files from UPX processing, there are cases where the UPX excludes need to be specified manually. For example, 32-bit Windows binaries from the PySide2 package (Qt5 DLLs and python extension modules) have been reported to be corrupted by UPX.

Changed in version 5.0: Unlike earlier releases that compared the provided UPX-exclude names against basenames of the collect binary files (and, due to incomplete case normalization, required provided exclude names to be lowercase on Windows), the UPX-exclude pattern matching now uses OS-default case sensitivity and supports the wildcard (*) operator. It also supports specifying (full or partial) parent path of the file.

The provided UPX exclude patterns are matched against source (origin) paths of the collected binary files, and the matching is performed from right to left.

For example, to exclude Qt5 DLLs from the PySide2 package, use --upx-exclude "Qt*.dll", and to exclude the python extensions from the PySide2 package, use --upx-exclude "PySide2\*.pyd".

Splash Screen (Experimental)
Note

This feature is incompatible with macOS. In the current design, the splash screen operates in a secondary thread, which is disallowed by the Tcl/Tk (or rather, the underlying GUI toolkit) on macOS.

Some applications may require a splash screen as soon as the application (bootloader) has been started, because especially in onefile mode large applications may have long extraction/startup times, while the bootloader prepares everything, where the user cannot judge whether the application was started successfully or not.

The bootloader is able to display a one-image (i.e. only an image) splash screen, which is displayed before the actual main extraction process starts. The splash screen supports non-transparent and hard-cut-transparent images as background image, so non-rectangular splash screens can also be displayed.

Note

Splash images with transparent regions are not supported on Linux due to Tcl/Tk platform limitations. The -transparentcolor and -transparent wm attributes used by PyInstaller are not available to Linux.

This splash screen is based on Tcl/Tk, which is the same library used by the Python module tkinter. PyInstaller bundles the dynamic libraries of tcl and tk into the application at compile time. These are loaded into the bootloader at startup of the application after they have been extracted (if the program has been packaged as an onefile archive). Since the file sizes of the necessary dynamic libraries are very small, there is almost no delay between the start of the application and the splash screen. The compressed size of the files necessary for the splash screen is about 1.5 MB.

As an additional feature, text can optionally be displayed on the splash screen. This can be changed/updated from within Python. This offers the possibility to display the splash screen during longer startup procedures of a Python program (e.g. waiting for a network response or loading large files into memory). You can also start a GUI behind the splash screen, and only after it is completely initialized the splash screen can be closed. Optionally, the font, color and size of the text can be set. However, the font must be installed on the user system, as it is not bundled. If the font is not available, a fallback font is used.

If the splash screen is configured to show text, it will automatically (as onefile archive) display the name of the file that is currently being unpacked, this acts as a progress bar.

The pyi_splash Module
The splash screen is controlled from within Python by the pyi_splash module, which can be imported at runtime. This module cannot be installed by a package manager because it is part of PyInstaller and is included as needed. This module must be imported within the Python program. The usage is as follows:

import pyi_splash

# Update the text on the splash screen
pyi_splash.update_text("PyInstaller is a great software!")
pyi_splash.update_text("Second time's a charm!")

# Close the splash screen. It does not matter when the call
# to this function is made, the splash screen remains open until
# this function is called or the Python program is terminated.
pyi_splash.close()
Of course the import should be in a try ... except block, in case the program is used externally as a normal Python script, without a bootloader. For a detailed description see pyi_splash Module (Detailed).

Defining the Extraction Location
When building your application in onefile mode (see Bundling to One File and How the One-File Program Works), you might encounter situations where you want to control the location of the temporary directory where the application unpacks itself. For example:

your application is supposed to be running for long periods of time, and you need to prevent its files from being deleted by the OS that performs periodic clean-up in standard temporary directories.

your target POSIX system does not use standard temporary directory location (i.e., /tmp) and the standard environment variables for temporary directory are not set in the environment.

the default temporary directory on the target POSIX system is mounted with noexec option, which prevents the frozen application from loading the unpacked shared libraries.

The location of the temporary directory can be overridden dynamically, by setting corresponding environment variable(s) before launching the application, or set statically, using the --runtime-tmpdir option during the build process.

Using environment variables
The extraction location can be controlled dynamically, by setting the environment variable(s) that PyInstaller uses to determine the temporary directory. This can, for example, be done in a wrapper shell script that sets the environment variable(s) before running the frozen application’s executable.

On POSIX systems, the environment variables used for temporary directory location are TMPDIR, TEMP, and TMP, in that order; if none are defined (or the corresponding directories do not exist or cannot be used), /tmp, /var/tmp, and /usr/tmp are used as hard-coded fall-backs, in the specified order. The directory specified via the environment variable must exist (i.e., the application attempts to create only its own directory under the base temporary directory).

On Windows, the default temporary directory location is determined via GetTempPathW function (which looks at TMP and TEMP environment variables for initial temporary directory candidates).

Using the --runtime-tmpdir option
The location of the temporary directory can be set statically, at compile time, using the --runtime-tmpdir option. If this option is used, the bootloader will ignore temporary directory locations defined by the OS, and use the specified path. The path can be either absolute or relative (which makes it relative to the current working directory).

Please use this option only if you know what you are doing.

Note

On POSIX systems, PyInstaller’s bootloader does not perform shell-style environment variable expansion on the path string given via --runtime-tmpdir option. Therefore, using environment variables (e.g., ~ or $HOME) in the path will not work.

Supporting Multiple Platforms
If you distribute your application for only one combination of OS and Python, just install PyInstaller like any other package and use it in your normal development setup.

Supporting Multiple Python Environments
When you need to bundle your application within one OS but for different versions of Python and support libraries – for example, a Python 3.6 version and a Python 3.7 version; or a supported version that uses Qt4 and a development version that uses Qt5 – we recommend you use venv. With venv you can maintain different combinations of Python and installed packages, and switch from one combination to another easily. These are called virtual environments or venvs in short.

Use venv to create as many different development environments as you need, each with its unique combination of Python and installed packages.

Install PyInstaller in each virtual environment.

Use PyInstaller to build your application in each virtual environment.

Note that when using venv, the path to the PyInstaller commands is:

Windows: ENV_ROOT\Scripts

Others: ENV_ROOT/bin

Under Windows, the pip-Win package makes it especially easy to set up different environments and switch between them. Under GNU/Linux and macOS, you switch environments at the command line.

See PEP 405 and the official Python Tutorial on Virtual Environments and Packages for more information about Python virtual environments.

Supporting Multiple Operating Systems
If you need to distribute your application for more than one OS, for example both Windows and macOS, you must install PyInstaller on each platform and bundle your app separately on each.

You can do this from a single machine using virtualization. The free virtualBox or the paid VMWare and Parallels allow you to run another complete operating system as a “guest”. You set up a virtual machine for each “guest” OS. In it you install Python, the support packages your application needs, and PyInstaller.

A File Sync & Share system like NextCloud is useful with virtual machines. Install the synchronization client in each virtual machine, all linked to your synchronization account. Keep a single copy of your script(s) in a synchronized folder. Then on any virtual machine you can run PyInstaller thus:

cd ~/NextCloud/project_folder/src # GNU/Linux, Mac -- Windows similar
rm *.pyc # get rid of modules compiled by another Python
pyinstaller --workpath=path-to-local-temp-folder  \
            --distpath=path-to-local-dist-folder  \
            ...other options as required...       \
            ./myscript.py
PyInstaller reads scripts from the common synchronized folder, but writes its work files and the bundled app in folders that are local to the virtual machine.

If you share the same home directory on multiple platforms, for example GNU/Linux and macOS, you will need to set the PYINSTALLER_CONFIG_DIR environment variable to different values on each platform otherwise PyInstaller may cache files for one platform and use them on the other platform, as by default it uses a subdirectory of your home directory as its cache location.

It is said to be possible to cross-develop for Windows under GNU/Linux using the free Wine environment. Further details are needed, see How to Contribute.

Capturing Windows Version Data
A Windows app may require a Version resource file. A Version resource contains a group of data structures, some containing binary integers and some containing strings, that describe the properties of the executable. For details see the Microsoft Version Information Structures page.

Version resources are complex and some elements are optional, others required. When you view the version tab of a Properties dialog, there’s no simple relationship between the data displayed and the structure of the resource. For this reason PyInstaller includes the pyi-grab_version command. It is invoked with the full path name of any Windows executable that has a Version resource:

pyi-grab_version executable_with_version_resource

The command writes text that represents a Version resource in readable form to standard output. You can copy it from the console window or redirect it to a file. Then you can edit the version information to adapt it to your program. Using pyi-grab_version you can find an executable that displays the kind of information you want, copy its resource data, and modify it to suit your package.

The version text file is encoded UTF-8 and may contain non-ASCII characters. (Unicode characters are allowed in Version resource string fields.) Be sure to edit and save the text file in UTF-8 unless you are certain it contains only ASCII string values.

Your edited version text file can be given with the --version-file option to pyinstaller or pyi-makespec. The text data is converted to a Version resource and installed in the bundled app.

In a Version resource there are two 64-bit binary values, FileVersion and ProductVersion. In the version text file these are given as four-element tuples, for example:

filevers=(2, 0, 4, 0),
prodvers=(2, 0, 4, 0),
The elements of each tuple represent 16-bit values from most-significant to least-significant. For example the value (2, 0, 4, 0) resolves to 0002000000040000 in hex.

You can also install a Version resource from a text file after the bundled app has been created, using the pyi-set_version command:

pyi-set_version version_text_file executable_file

The pyi-set_version utility reads a version text file as written by pyi-grab_version, converts it to a Version resource, and installs that resource in the executable_file specified.

For advanced uses, examine a version text file as written by pyi-grab_version. You find it is Python code that creates a VSVersionInfo object. The class definition for VSVersionInfo is found in utils/win32/versioninfo.py in the PyInstaller distribution folder. You can write a program that imports versioninfo. In that program you can eval the contents of a version info text file to produce a VSVersionInfo object. You can use the .toRaw() method of that object to produce a Version resource in binary form. Or you can apply the unicode() function to the object to reproduce the version text file.

Building macOS App Bundles
Under macOS, PyInstaller always builds a UNIX executable in dist. If you specify --onedir, the output is a folder named myscript containing supporting files and an executable named myscript. If you specify --onefile, the output is a single UNIX executable named myscript. Either executable can be started from a Terminal command line. Standard input and output work as normal through that Terminal window.

If you specify --windowed with either option, the dist folder also contains a macOS app bundle named myscript.app.

Note

Generating app bundles with onefile executables (i.e., using the combination of --onefile and --windowed options), while possible, is not recommended. Such app bundles are inefficient, because they require unpacking on each run (and the unpacked content might be scanned by the OS each time). Furthermore, onefile executables will not work when signed/notarized with sandbox enabled (which is a requirement for distribution of apps through Mac App Store).

As you are likely aware, an app bundle is a special type of folder. The one built by PyInstaller always contains a folder named Contents, which contains:

A file named Info.plist that describes the app.

A folder named MacOS that contains the program executable.

A folder named Frameworks that contains the collected binaries (shared libraries, python extensions) and nested .framework bundles. It also contains symbolic links to data files and directories from the Resources directory.

A folder named Resources that contains the icon file and all collected data files. It also contains symbolic links to binaries and directories from the Resources directory.

Note

The contents of the Frameworks and Resources directories are cross-linked between the two directories in an effort to maintain an illusion of a single content directory (which is required by some packages), while also trying to satisfy the Apple’s file placement requirements for codesigning.

Use the --icon argument to specify a custom icon for the application. It will be copied into the Resources folder. (If you do not specify an icon file, PyInstaller supplies a file icon-windowed.icns with the PyInstaller logo.)

Use the --osx-bundle-identifier argument to add a bundle identifier. This becomes the CFBundleIdentifier used in code-signing (see the PyInstaller code signing recipe and for more detail, the Apple code signing overview technical note).

You can add other items to the Info.plist by editing the spec file; see Spec File Options for a macOS Bundle below.

Platform-specific Notes
GNU/Linux
Making GNU/Linux Apps Forward-Compatible
Under GNU/Linux, PyInstaller does not bundle libc (the C standard library, usually glibc, the Gnu version) with the app. Instead, the app expects to link dynamically to the libc from the local OS where it runs. The interface between any app and libc is forward compatible to newer releases, but it is not backward compatible to older releases.

For this reason, if you bundle your app on the current version of GNU/Linux, it may fail to execute (typically with a runtime dynamic link error) if it is executed on an older version of GNU/Linux.

The solution is to always build your app on the oldest version of GNU/Linux you mean to support. It should continue to work with the libc found on newer versions.

The GNU/Linux standard libraries such as glibc are distributed in 64-bit and 32-bit versions, and these are not compatible. As a result you cannot bundle your app on a 32-bit system and run it on a 64-bit installation, nor vice-versa. You must make a unique version of the app for each word-length supported.

Note that PyInstaller does bundle other shared libraries that are discovered via dependency analysis, such as libstdc++.so.6, libfontconfig.so.1, libfreetype.so.6. These libraries may be required on systems where older (and thus incompatible) versions of these libraries are available. On the other hand, the bundled libraries may cause issues when trying to load a system-provided shared library that is linked against a newer version of the system-provided library.

For example, system-installed mesa DRI drivers (e.g., radeonsi_dri.so) depend on the system-provided version of libstdc++.so.6. If the frozen application bundles an older version of libstdc++.so.6 (as collected from the build system), this will likely cause missing symbol errors and prevent the DRI drivers from loading. In this case, the bundled libstdc++.so.6 should be removed. However, this may not work on a different distribution that provides libstdc++.so.6 older than the one from the build system; in that case, the bundled version should be kept, because the system-provided version may lack the symbols required by other collected binaries that depend on libstdc++.so.6.

Windows
The developer needs to take special care to include the Visual C++ run-time .dlls: Python 3.5+ uses Visual Studio 2015 run-time, which has been renamed into “Universal CRT“ and has become part of Windows 10. For Windows Vista through Windows 8.1 there are Windows Update packages, which may or may not be installed in the target-system. So you have the following options:

Build on Windows 7 which has been reported to work.

Include one of the VCRedist packages (the redistributable package files) into your application’s installer. This is Microsoft’s recommended way, see “Distributing Software that uses the Universal CRT“ in the above-mentioned link, numbers 2 and 3.

Install the Windows Software Development Kit (SDK) for Windows 10 and expand the .spec-file to include the required DLLs, see “Distributing Software that uses the Universal CRT“ in the above-mentioned link, number 6.

If you think, PyInstaller should do this by itself, please help improving PyInstaller.

macOS
Making macOS apps Forward-Compatible
On macOS, system components from one version of the OS are usually compatible with later versions, but they may not work with earlier versions. While PyInstaller does not collect system components of the OS, the collected 3rd party binaries (e.g., python extension modules) are built against specific version of the OS libraries, and may or may not support older OS versions.

As such, the only way to ensure that your frozen application supports an older version of the OS is to freeze it on the oldest version of the OS that you wish to support. This applies especially when building with Homebrew python, as its binaries usually explicitly target the running OS.

For example, to ensure compatibility with “Mojave” (10.14) and later versions, you should set up a full environment (i.e., install python, PyInstaller, your application’s code, and all its dependencies) in a copy of macOS 10.14, using a virtual machine if necessary. Then use PyInstaller to freeze your application in that environment; the generated frozen application should be compatible with that and later versions of macOS.

Building 32-bit Apps in macOS
Note

This section is largely obsolete, as support for 32-bit application was removed in macOS 10.15 Catalina (for 64-bit multi-arch support on modern versions of macOS, see here). However, PyInstaller still supports building 32-bit bootloader, and 32-bit/64-bit Python installers are still available from python.org for (some) versions of Python 3.7 which PyInstaller dropped support for in v6.0.

Older versions of macOS supported both 32-bit and 64-bit executables. PyInstaller builds an app using the the word-length of the Python used to execute it. That will typically be a 64-bit version of Python, resulting in a 64-bit executable. To create a 32-bit executable, run PyInstaller under a 32-bit Python.

To verify that the installed python version supports execution in either 64- or 32-bit mode, use the file command on the Python executable:

$ file /usr/local/bin/python3
/usr/local/bin/python3: Mach-O universal binary with 2 architectures
/usr/local/bin/python3 (for architecture i386):     Mach-O executable i386
/usr/local/bin/python3 (for architecture x86_64):   Mach-O 64-bit executable x86_64
The OS chooses which architecture to run, and typically defaults to 64-bit. You can force the use of either architecture by name using the arch command:

$ /usr/local/bin/python3
Python 3.7.6 (v3.7.6:43364a7ae0, Dec 18 2019, 14:12:53)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys; sys.maxsize
9223372036854775807

$ arch -i386 /usr/local/bin/python3
Python 3.7.6 (v3.7.6:43364a7ae0, Dec 18 2019, 14:12:53)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys; sys.maxsize
2147483647
Note

PyInstaller does not provide pre-built 32-bit bootloaders for macOS anymore. In order to use PyInstaller with 32-bit python, you need to build the bootloader yourself, using an XCode version that still supports compiling 32-bit. Depending on the compiler/toolchain, you may also need to explicitly pass --target-arch=32bit to the waf command.

Getting the Opened Document Names
When user double-clicks a document of a type that is registered with your application, or when a user drags a document and drops it on your application’s icon, macOS launches your application and provides the name(s) of the opened document(s) in the form of an OpenDocument AppleEvent.

These events are typically handled via installed event handlers in your application (e.g., using Carbon API via ctypes, or using facilities provided by UI toolkits, such as tkinter or PyQt5).

Alternatively, PyInstaller also supports conversion of open document/URL events into arguments that are appended to sys.argv. This applies only to events received during application launch, i.e., before your frozen code is started. To handle events that are dispatched while your application is already running, you need to set up corresponding event handlers.

For details, see this section.