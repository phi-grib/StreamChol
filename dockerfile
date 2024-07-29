FROM python:3.8

EXPOSE 8501

ADD streamchol.py .

ADD requeriments.txt .

ADD juntas.png .

ADD model_bcrp.pkl .

ADD model_bsep.pkl .

ADD model_mrp2.pkl .

ADD model_mrp3.pkl .

ADD model_mrp4.pkl .

ADD model_oat1.pkl .

ADD model_oat2.pkl .

ADD model_pgp.pkl .

ADD final_predictions_app.csv .

ADD drawi3.svg .

RUN apt-get --force-yes update \
    && apt-get --assume-yes install r-base-core

RUN R -e "install.packages(c('dplyr', 'httk','stringr'))"


RUN pip install -r requeriments.txt

#RUN pip install protobuf==3.20 altair==4.0


ENTRYPOINT ["streamlit", "run","--server.fileWatcherType", "None"]
CMD ["streamchol.py"]