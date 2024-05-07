# StreamChol


This is a web application for predicting drug-like compounds that may induce cholestasis. The methodology is based on the article we published: [https://pubs.acs.org/doi/10.1021/acs.jcim.3c00945](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00945).

<p align="center">
  <img src="https://github.com/phi-grib/StreamChol/blob/main/cover%20page.PNG" alt="Cover Page">
</p>


# StreamChol in Docker hub
A docker container (https://www.docker.com/), fully configured can be downloaded from DockerHub and installed using:

docker run -d -p 8501:8501 parodbe/streamchol_second_version

Then, the StreamChol will be accesible from a web browser at address http://localhost:8501
