# docker

This dockerfile is part of a much larger project for the Dutch national police and is part of my graduation project. The aim of the project was to creat timelines of suspects based on analysis of live cctv footage. This dockerfile created the image acquisition service and was responsible for reading and preprocessing frames acquired from ip camera's and sending them to the analysis service via an apache kafka message broker. As it is only the dockerfile and misses the python project it will not build at the moment.

As it uses opencv as dependency the image got very large, I solved this by using a two stage build that reduced the image size by about 500MB