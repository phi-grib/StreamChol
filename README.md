# StreamChol


This is a web application for predicting drug-like compounds that may induce cholestasis. The methodology is based on the article we published: [https://pubs.acs.org/doi/10.1021/acs.jcim.3c00945](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00945).

<p align="center">
  <img src="https://github.com/phi-grib/StreamChol/blob/main/cover%20page.PNG" alt="Cover Page">
</p>

# Files explanation

Components for building the application:

-	The application itself (streamchol.py)

-	The pickle files for the individual models

-	Requirements.txt

-	Dockerfile

Additional files for testing:

-	Training series (final_predictions_app.csv)

-	A test set (test.xlsx)


# StreamChol in Docker hub
A docker container (https://www.docker.com/), fully configured can be downloaded from DockerHub and installed using:

```
docker run -d -p 8501:8501 parodbe/streamchol_app
```

Then, the StreamChol will be accesible from a web browser at address http://localhost:8501


# Stand-alone Deployment

These steps will guide you through running the app directly on your local machine without Docker.

### Prerequisites
Ensure your system meets the following requirements:
- **R**: Version 4.0 or higher
- **Rtools** (Windows only)
- **Microsoft Build Tools** (Windows only)
- **Python**: Version 3.8 or higher (if applicable)
- Properly configured **environment variables** (details below)

---

### Installation Steps

#### **Step 1: Install R**
1. Download the latest version of R from [CRAN](https://cran.r-project.org/).
2. Install R:
   - **macOS**: Drag the R package to the Applications folder.
   - **Windows**: Run the `.exe` installer and follow the on-screen instructions.
   - **Linux**: Use your package manager:
     ```bash
     sudo apt update
     sudo apt install r-base
     ```
3. Verify the installation:
   ```bash
   R --version
   ```

#### **Step 2: Install Rtools (Windows Only)**
1. Download **Rtools** from [Rtools for Windows](https://cran.r-project.org/bin/windows/Rtools/).
2. Install Rtools and ensure that its path is added to your system environment variables.
3. Verify the installation:
   ```bash
   R CMD config --ldflags
   ```

#### **Step 3: Install Microsoft Build Tools (Windows Only)**
1. Download and install **Microsoft Build Tools** from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. During installation, select the following:
   - **Workload**: "Desktop Development with C++".
   - **Individual Components**:
     - MSVC Compiler
     - Windows 10/11 SDK
     - CMake tools (optional but recommended)
3. Verify the installation:
   ```bash
   cl
   ```

---

### 2. Configuring Environment Variables

#### **For R**
1. Add the R installation directory to your `PATH`:
   - Example:
     ```
     C:\Program Files\R\R-4.4.2\bin\x64
     ```
2. Verify:
   ```bash
   R --version
   ```

#### **For Rtools**
1. Add the Rtools directory to your `PATH`:
   - Example for Rtools44:
     ```
     C:\rtools44\usr\bin
     C:\rtools44\x86_64-w64-mingw32.static.posix\bin
     ```
2. Verify:
   ```bash
   gcc --version
   ```

#### **For Microsoft Build Tools**
1. Add the directory containing `cl.exe` to your `PATH`:
   - Example:
     ```
     C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64
     ```
2. Verify:
   ```bash
   cl
   ```

---

### 3. Install Required Libraries

#### **For R**
Run the following in an R console:
```R
install.packages(c("httk", "dplyr", "stringr"))
# Add any other packages your app requires
```

#### **For Python**
If your app uses Python, install the required libraries:
1. Install Python:
   ```bash
   python --version
   ```
   Make sure Python is in your `PATH`.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### 4. Run the application

1. To activate in the app.py file the following steps to set the R environment:

   ```bash
   os.environ["R_HOME"] = 'C:/Program Files/R/R-4.4.12' 
   os.environ["PATH"] = 'C:/Program Files/R/R-4.4.2/bin/x64' + ";" + os.environ["PATH"]
   ``` 
2. Run streamlit app.py
   ```
   http://localhost:8501
   ```

