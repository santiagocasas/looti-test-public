 #The line below determines the build image to use
FROM tensorflow/tensorflow:latest

#The next block determines what dependencies to load
RUN pip install pandas
RUN pip install numpy
RUN pip install scipy
RUN pip install future
RUN pip install matplotlib
RUN pip install sklearn
RUN apt-get update
RUN apt-get install -y git

RUN pip install git+https://github.com/santiagocasas/looti-test-public.git
# run as the user "galileo" with associated working directory
RUN useradd -ms /bin/bash galileo
USER galileo
WORKDIR /home/galileo
#This line determines where to copy project files from, and where to copy them to
COPY . .

#The entrypoint is the command used to start your project
ENTRYPOINT ["python3","./MassiveNus_validation_script.py"]
