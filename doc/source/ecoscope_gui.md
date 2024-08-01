Ecoscope GUI
----

## Features

The Ecoscope GUI (graphical user interface) is an easy-to-use cross-platform software that allows you to:

- choose events or subject group observations from your EarthRanger instance, optionally filtered by date ranges
- download the data
- export the resulting data as either a geopackage (`.gpkg`) or a CSV (`.csv`) file and save it to your machine.

The GUI supports 3 user interface languages at the moment:

- 🇬🇧 English
- 🇫🇷 French
- 🇪🇸 Spanish

Screenshots can be found at [the bottom of the page](#Screenshots).

The GUI uses `ecoscope` under the hood to filter, clean, and process the downloaded data. The interface is built with
Python and the Qt graphical framework.

## Requirements

Please review the system requirements below that are needed to run the software before proceeding to
the [Downloads](#Downloads) section.

| OS      | CPU architecture       | Free Disk Space | Notes                                                         |
|---------|------------------------|-----------------|---------------------------------------------------------------|
| Windows | 64-bit                 | at least 1.5 GB | We've tested the software on Windows 11                       |
| MacOS   | Apple Silicon (64-bit) | at least 500 MB | We've tested the software on MacOS Monterey, MacOS Ventura, and MacOS Sonoma |
| MacOS   | Intel (64-bit)         | at least 1.5 GB | We've tested the software on MacOS Monterey, MacOS Ventura, and MacOS Sonoma |
| Linux   | 64-bit                 | at least 1.5 GB | We've tested the software on Ubuntu 22.04                     |

## Downloads

**We are actively working on open-sourcing the GUI code but until then downloads are hosted on the EarthRanger team's
Google Drive (links below).** Once the code has been open-sourced, releases will be provided via GitHub. More details
will
be provided on this page when that happens.

:::{note}
We do not collect any usage or other data about your use of the software. In addition, to ensure the safety of your
EarthRanger account, your login credentials are never saved to disk in plaintext. Instead, we will ask you to re-enter
your password
to confirm your identity every time you open the software.
:::

:::{tip}
For security-conscious users, as long as we host the files on Google Drive, we will provide the SHA-256 download
checksums below, so you can verify the integrity of the files you download.
:::

-----------

### Windows

:::{important}
Please make sure to review the instructions **before starting your download**.
:::

#### Instructions

1. Download the `.exe` file from the link below
2. Right-click on the downloaded `.exe` file and choose `Run as Administrator`
3. Follow the prompts until the installation succeeds
4. You will now have shortcuts on your Windows desktop and in the Windows start menu to start the program or to
   uninstall it

| Link                                                                          | Download Size (approximate) | Install Size (approximate) | SHA-256 checksum                                                   |
|-------------------------------------------------------------------------------|-----------------------------|----------------------------|--------------------------------------------------------------------|
| [Download](https://drive.google.com/file/d/1Pgnbvpv-_B3kyXpKAEPWhHmIP17VIyNQ) | 200 MB                      | 930 MB                     | `2bd107ce2e2d6ff7544f52a659622b813b84291bcc66688ffd0f167eac26d8ad` |

-----------

### MacOS

:::{important}
Please make sure to review the instructions **before starting your download**.
:::

#### Instructions

1. Download the `.zip` file from the link below
2. Unzip the downloaded `.zip` file to get an `Ecoscope GUI.app` file
3. Open a command-prompt and type the following command:

```
xattr -dr com.apple.quarantine '/path/to/Ecoscope GUI.app'
```

For example, if you downloaded the file to your `Downloads` folder, you would write

```
xattr -dr com.apple.quarantine '/Users/your_macos_username/Downloads/Ecoscope GUI.app'
```

4. Double-click on the `Ecoscope GUI.app` file in Finder to run it.

| Link                                                                                              | Download Size (approximate) | Install Size (approximate) | SHA-256 checksum                                                   |
|---------------------------------------------------------------------------------------------------|-----------------------------|----------------------------|--------------------------------------------------------------------|
| (**Apple Silicon**) [Download](https://drive.google.com/file/d/1phXHa6V1m20_ZWJlNm8LGG8GWXC7IrRm) | 170 MB                      | 470 MB                     | `62de33715266c30c2b873a989547c17d2e92858d9ad4c0becc118fd68a124bcc` |
| (**Intel**) [Download](https://drive.google.com/file/d/1EWh-V7iM6CYMNLa5fHm9a0Rj-OCCIXiU)         | 410 MB                      | 1.35 GB                    | `3edd0d5578f744eb415b26634a71a1bbea1a7c898991dfdfa67ca311e169a968` |

:::{attention}
Step 3 above is mandatory. If you skip it, you will not be able to open the program by
double-clicking `Ecoscope GUI.app`. Instead, you will see the following error dialog that
says `"Ecoscope GUI.app" is damaged and can't be opened.`

   ```{figure-md}
   ![GUI Interface](_static/images/gui_interface_damaged_error.png){.bg-primary .mb-1 width=100px}

   '"Ecoscope GUI.app" is damaged and can't be opened' error.
   ```

If you see this error, do Step 3 and try double-clicking on `Ecoscope GUI.app` again. You do not need to re-download the
app.
:::

:::{note}
We're working on improving this so that you don't have to do step 3.
:::
-----------

### Linux

:::{important}
Please make sure to review the instructions **before starting your download**.
:::

#### Instructions

1. Download the `.zip` file from the link below.
2. Unzip the download `.zip` file to get an `Ecoscope GUI` directory.
3. Go into the `Ecoscope GUI` directory.
4. Double-click on the `Ecoscope GUI` binary executable.

| Link                                                                          | Download Size (approximate) | Install Size (approximate) | SHA-256 checksum                                                   |
|-------------------------------------------------------------------------------|-----------------------------|----------------------------|--------------------------------------------------------------------|
| [Download](https://drive.google.com/file/d/14dWdXmR5EsOiLmIR3PJFtZVZQUSE410K) | 566 MB                      | 1.7 GB                     | `e8d7b352d3276d8a325105bc6fbe6d3ebfa156f3d3d0a09b900b6aa18a96e8fd` |

## Screenshots

:::{figure-md}
![GUI Interface](_static/images/gui_interface_1.png){.bg-primary .mb-1 width=200px}

Events download configuration screen (English)
:::

:::{figure-md}
![GUI Interface](_static/images/gui_interface_2.png){.bg-primary .mb-1 width=200px}

Subject group observations download configuration screen (French)
:::

:::{figure-md}
![GUI Interface](_static/images/gui_interface_3.png){.bg-primary .mb-1 width=200px}

Confirm your password on startup screen (Spanish)
:::
