FROM python:3.8

EXPOSE 8501

ADD app_streamchol.py .

ADD requeriments.txt .

ADD juntas.png .

ADD chembl_data_bcrp.sdf .

ADD chembl_data_bsep.sdf .

ADD chembl_data_mrp2.sdf .

ADD chembl_data_mrp3.sdf .

ADD chembl_data_mrp4.sdf .

ADD chembl_data_OATP1b1.sdf .

ADD chembl_data_OATP1b3.sdf .

ADD chembl_data_pgp.sdf .

ADD final_predictions_app.csv .

ADD drawi3.svg .

RUN apt-get --force-yes update \
    && apt-get --assume-yes install r-base-core

RUN R -e "install.packages(c('dplyr', 'httk','stringr'))"


RUN pip install -r requeriments.txt

#RUN pip install protobuf==3.20 altair==4.0


ENTRYPOINT ["streamlit", "run","--server.fileWatcherType", "None"]
CMD ["app_streamchol.py"]
