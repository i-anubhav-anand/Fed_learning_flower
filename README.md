# Federated_learning_flower


<h1>Introduction</h1>
How Flower can be used to build federated learning use cases based on existing machine learning projects.

<h2>Project Setup</h2>

containing files:
```bash

dataset.zip
   |__covid19
        |__ *.jpg (all image files)
   |__normal
         |__ *.jpg (all image files)
 --client.py
 --client2.py
 -- server.py
 -- README.md
 --requiremtns.txt
```

 
  
  <h1>Hosting server.py on cloud(EC2) and adding port 8080 in security group.</h1>
  1.Log into the AWS Management Console.
  
  2.Scroll down the left navigation panel and choose "Security Group" under "Network & Security".
   
   ![image](https://user-images.githubusercontent.com/76263415/174425642-3f7d086a-1786-436b-a992-7ec4f1dbe087.png)
   
  3.Select the "EC2 Security Group" that needs to be verified.
  
  ![image](https://user-images.githubusercontent.com/76263415/174425658-56404bde-a5de-448d-bae1-c8ea8c47f9f2.png)

  4.Scroll down the bottom panel and choose "Inbound". Choose "Custom TCP rule" in the dropdown.Then you will be able to change the port to 8080.
   ![inbound](https://user-images.githubusercontent.com/76263415/174425457-bbc38b00-4534-47d8-b28d-0daf3e57958d.png)
  5.Click on the "Save" button to make the necessary changes.
  
  ![image](https://user-images.githubusercontent.com/76263415/174425684-675f2fab-6cc8-4049-94ac-61f033038975.png)
  
  
  Now, Connect to your EC2 instance.
  
  <h1>Clone the contents of the Repo into this directory using SSH.</h1>
  (Note: The ‘.’ at the end of the command is to put the contents of the repository into the current directory)
  
```bash

$ git clone https://github.com/i-anubhav-anand/Fed_learning_flower

```
     
   Go to the (Fed_learning_flower) folder and install all the dependencies

```bash
    $ pip install -r requirements.txt.
```
  After installation run 
  
```bash
    $ python3 server.py
```
  Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process.
  To do so simply open two more   terminal windows and run the following commands.


 <h1>From Centralized To Federated </h1>
 
  This Project is based on the Deep Learning with PyTorch that uses the custom data of Coivd and Normal Patient to predict if the patient is having covid or not.
  
  <h1>Setting up to run client on your local system</h1>
  
  You can simply start the centralized training as described
  
  Start by cloning the example project.

```bash

$ git clone https://github.com/i-anubhav-anand/Fed_learning_flower

```
  
  
  Create virtual env.
  
  ```bash
    $ python -m venv venv
 ```
 Activate virtual env.
 ```bash
$ # Linux/macOS
$ source venv/bin/activate  
$ # Windows
$ venv\Scripts\activate 
 ```
 Installing all the dependencies
```bash
    $ pip install -r requirements.txt.
```
Before running client.py/client2.py make sure to replace localhost with the  public ip address of EC2 instance.
 
 ![image](https://user-images.githubusercontent.com/76263415/174426161-751c0702-bbd8-4fac-af7b-2d1e3f92f5d0.png)

which looks like this

![image](https://user-images.githubusercontent.com/76263415/174426355-ab394c55-fe69-4810-b3cf-18cdf784a7d0.png)



Start client 1 in the first terminal:

  ```bash
    $ python client.py
```

  Start client 2 in the second terminal:

```bash
    $ python client2.py
```  




  
  
  You are now training a PyTorch-based CNN image classifier on Chest X-ray(Custom Data), federated across two clients.
  


 <h1>Snapshot</h1>

![image](https://user-images.githubusercontent.com/76263415/174462612-7be7c454-0104-4b8a-a36d-756d7c9f28ce.png)


<h1> Working Demo </h1>

https://user-images.githubusercontent.com/76263415/174444042-2a4c0aee-0da4-4132-9534-a8705c95785f.mp4


<h1> Single/Batch Prediction </h1>

Once after you're done with the training the path weights get saved locally 

To run single/Batch Prediction make sure you pass the right arguments


```bash
    $ python test.py arg1 arg2 
```  

**where arg1 take '1' for Single Image Prediction and '2' for Batch Prediction arg2 which is the location the image/dir respectively**

<h3>
   For Example </h3>
   
   
 <h3>  For Single Image Prediction</h3>

```bash
    $ python .\covid_prediction.py 1 'dataset\dataset\covid19\person3_bacteria_13.jpeg'
```  
<h1>Output</h1>

![image](https://user-images.githubusercontent.com/76263415/174462789-07b0bdfb-ec01-493f-81fc-2d92a8c4f35a.png)



<h3>For Batch Prediction</h3>

```bash
    $ python .\covid_prediction.py 2 'dataset\dataset'   
```  

<h1>Output</h1>

![image](https://user-images.githubusercontent.com/76263415/174462035-59f503fb-4ae4-42e1-be5c-0014acd5dbeb.png)


<h1>Working Demo of Predicition</h1>


https://user-images.githubusercontent.com/76263415/174462441-26cf8ea4-ef65-4f6a-b2ec-8a515e195bcb.mp4






  
