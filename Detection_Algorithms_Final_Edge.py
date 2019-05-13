# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:56:14 2019
@author: TOM whn in SEW
"""
# importing needed libraries
from exchangelib import DELEGATE, Account, Credentials, FileAttachment, HTMLBody, Message,Mailbox, EWSDateTime,EWSTimeZone
from exchangelib.protocol import BaseProtocol, NoVerifyHTTPAdapter
import ConfigParser,sys,os,cv2,urllib3,numpy as np
from PIL import Image, ImageFile , ImageOps
from datetime import datetime, timedelta 
from time import sleep
from skimage.measure import compare_mse as mse

#Establishing connection
BaseProtocol.HTTP_ADAPTER_CLS = NoVerifyHTTPAdapter
urllib3.disable_warnings()
credentials = Credentials(

    username='EXCHANGE\\sm_sewerci',  # Or myusername@example.com f     or O365

    password='EqyUQejBgMT32'
)
account = Account(primary_smtp_address='sewercameraimages@sew.com.au',credentials=credentials,

    autodiscover=True,

    access_type=DELEGATE
)

#Connectiong to configuration file
if len(sys.argv)==2:         
    config_file=sys.argv[1]
else:
    config_file= 'pixel_config1.txt'
config= ConfigParser.ConfigParser()
config.readfp(open (config_file))
to_recipients= config.get('configurations','to_recipients').split(",")

#Creating an instance of message class in Exchangelib to send the email
m = Message(
                account=account,
                subject='This is a Test Message (Sewer Wells Camera Images Status)',
                to_recipients=to_recipients
      )

ImageFile.LOAD_TRUNCATED_IMAGES = True
camdetails=[]

#Calculating the number of days 
def fetchdate (right_now,daystofetch):
    one_day_before= right_now - timedelta(days=daystofetch)
    return one_day_before

#Method to create folder dynamically
def createfolder(wdir,dirnames):
    foldernames=os.listdir(wdir)
    for i in dirnames:
        if i not in foldernames :           
            os.makedirs(os.path.join(wdir,i))


# Function to get the latest image to compare from the mailbox
def fetchmailattachments(datetofetch,right_now,wdir):   
     print "Fetching Attachements from mailbox"
     all_items = account.inbox.filter(datetime_received__range=(datetofetch, right_now),sender__contains='@sewl.com.au')          
     if all_items.count()==0:
         print "No mails for this period exiting from execution"
         sleep(5)
         sys.exit()
     for item in all_items:
         for attachment in item.attachments:
            if isinstance(attachment, FileAttachment):                  
                attachmentname= ''.join(e for e in attachment.name if e.isalnum())
                folderdecider= attachment.name.split('_')[0]
                try:
                    local_path = os.path.join(wdir,folderdecider,attachmentname+'.jpg')
                    with open(local_path, 'wb') as f:
                        f.write(attachment.content)
                        f.close()                    
                except:
                    print "Please add the camera "+folderdecider+ "to the config file to receive the images" 
                    sleep(2)                              
                    sys.exit()  


# Function to get ref images
def getRefImages(img_ref_path,refdetails,refdetailsarray,imageflag):
    print "Getting Reference Images for " + refdetails.split('_')[0]
    image_ref= os.path.join(img_ref_path,refdetails)
    try:
        if imageflag==0:
            img_ref_grey = ImageOps.grayscale(Image.open(image_ref))
            img_ref_colour= Image.open(image_ref)
            a= refdetailsarray[i+2]
            EX= a.split(',')[0]
            x1=EX.split(':')[0]
            x2= EX.split(':')[1]
            WY= a.split(',')[1]
            y1= WY.split(':')[0]
            y2= WY.split(':')[1]
            img_ref_cut_grey=img_ref_grey.crop((float(x1),float(y1),float(x2),float(y2)))
            img_ref_cut_colour=img_ref_colour.crop((float(x1),float(y1),float(x2),float(y2)))
            img_ref_cut_grey= np.asarray(img_ref_cut_grey)
            img_ref_cut_colour= np.asarray(img_ref_cut_colour)
            img_ref_grey=np.asarray(img_ref_grey)
            return img_ref_cut_grey, img_ref_cut_colour
        else:
            img_ref_grey_line = ImageOps.grayscale(Image.open(image_ref))
            img_ref_colour_line= Image.open(image_ref)
            return img_ref_grey_line, img_ref_colour_line

    except:
        print 'Either the reference file is missing in location or the path in configuration file not correct'
        sleep(2)
        sys.exit()


# Function to get latest images
def getLatestImages(wdir,foldername,refdetailsarray,imageflag):
    print "Getting Latest Images for "+ foldername
    foldernames=os.listdir(wdir)
    if (foldername in foldernames):
          filenames=os.listdir(os.path.join(wdir,foldername))
          filearray=np.asarray(filenames)
          filearraysorted= np.sort(filearray)
          n= filearraysorted.size
          
          try:
              if imageflag==0:
                  filename=os.path.join(wdir,foldername,filearraysorted[n-1])
                  img_latest_grey = ImageOps.grayscale(Image.open(filename))
                  img_latest_colour= Image.open(filename)
                  a= refdetailsarray[i+2]
                  EX= a.split(',')[0]
                  x1=EX.split(':')[0]
                  x2= EX.split(':')[1]
                  WY= a.split(',')[1]
                  y1= WY.split(':')[0]
                  y2= WY.split(':')[1]
                  img_latest_cut_grey=img_latest_grey.crop((float(x1),float(y1),float(x2),float(y2)))
                  img_latest_cut_colour=img_latest_colour.crop((float(x1),float(y1),float(x2),float(y2)))
                  img_latest_cut_grey= np.asarray(img_latest_cut_grey)
                  img_latest_cut_colour= np.asarray(img_latest_cut_colour)
                  img_latest_grey=np.asarray(img_latest_grey)
                  return img_latest_cut_grey, img_latest_cut_colour
              else:
                  filename=os.path.join(wdir,foldername,filearraysorted[n-1])
                  img_latest_grey_line = ImageOps.grayscale(Image.open(filename))
                  img_latest_colour_line= Image.open(filename)
                  return img_latest_grey_line,img_latest_colour_line
          except:
              print 'The latest file is missing in location or the configuration file is not correct'
              sleep(2)                              
              sys.exit()


# Calculating Mean square error
def meansquare(image_ref_cut_grey,image_latest_cut_grey):
    mse_none= mse (image_ref_cut_grey,image_latest_cut_grey)
    return mse_none


# Applying percent change in pixels concept. Image object must be converted into an array for cv2 manipulation   
def pixeldirt(image_ref_cut_grey,image_latest_cut_grey):
    th1 = cv2.adaptiveThreshold(image_ref_cut_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    Def_dirt_per_ref= ((th1.size) - (cv2.countNonZero(th1)))/ float(th1.size)
    th2 = cv2.adaptiveThreshold(image_latest_cut_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    Def_dirt_per_latest= ((th2.size) - (cv2.countNonZero(th2)))/ float(th2.size)
    per_change_pix= (Def_dirt_per_ref - Def_dirt_per_latest)/Def_dirt_per_ref
    per_change_pix=per_change_pix*100
    return per_change_pix


# Applying colour plane histogram comparisons
def histogram(image_ref_cut_colour,image_latest_cut_colour):
    image_ref_cut_colour=cv2.bilateralFilter(image_ref_cut_colour,9,100,100)
    image_latest_cut_colour=cv2.bilateralFilter(image_latest_cut_colour,9,100,100)
    red_ref= cv2.calcHist([image_ref_cut_colour],[2],None,[256],[0,255])
    green_ref= cv2.calcHist([image_ref_cut_colour],[1],None,[256],[0,255])
    blue_ref= cv2.calcHist([image_ref_cut_colour],[0],None,[256],[0,255])
    red_latest= cv2.calcHist([image_latest_cut_colour],[2],None,[256],[0,255])
    green_latest= cv2.calcHist([image_latest_cut_colour],[1],None,[256],[0,255])
    blue_latest= cv2.calcHist([image_latest_cut_colour],[0],None,[256],[0,255])
    red_value= cv2.compareHist (red_ref, red_latest,cv2.HISTCMP_CORREL  )
    green_value=cv2.compareHist (green_ref, green_latest,cv2.HISTCMP_CORREL  )
    blue_value=cv2.compareHist (blue_ref, blue_latest,cv2.HISTCMP_CORREL  )
    avg_similarity=  (red_value + green_value + blue_value)/3
    return avg_similarity

#Applying ORB Algorithm
#def orbalgorithm(image_ref_cut_grey,image_latest_cut_grey):
#    orb = cv2.ORB_create()
#    kp1, des1 = orb.detectAndCompute(image_ref_cut_grey,None)
#    kp2, des2 = orb.detectAndCompute(image_latest_cut_grey,None)
## Brute Force Matching
#    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
#    try:
#        matches=bf.match(des1,des2)
#        matches=sorted(matches,key=lambda x:x.distance)
#        return len(matches)
#    except: 
#        return 0
    
#Using Edge detection Technique.This function can be used at a later point of time if required with some fine tunings
#Change the configuration file to include the marked reference image
def edgedetection(image_ref_grey_line,image_latest_grey_line):
    #The reference grayscale image used here will be marked with areas of interest.
    simpix=0;  #Variable holds the number of same pixels coordinate with white pixels
    image_ref_grey_line=np.asarray(image_ref_grey_line)
    white_pix_ref = np.where(image_ref_grey_line== [255])
    white_pix_ref_c = zip(white_pix_ref[0], white_pix_ref[1])
    image_latest_grey_line=np.asarray(image_latest_grey_line)
    image_latest_grey_line=cv2.equalizeHist(image_latest_grey_line)
    image_latest_grey_line=cv2.bilateralFilter(image_latest_grey_line,9,100,100)
    edges=cv2.Canny(image_latest_grey_line,20,250)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    lines_latest = cv2.HoughLinesP(dilation, 1, np.pi/180, 100,minLineLength=120, maxLineGap=5)

    try:
        for line in lines_latest:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_latest_grey_line, (x1, y1), (x2, y2), (255, 255, 255), 2)
        white_pix_latest = np.where(image_latest_grey_line == [255])
        white_pix_latest_c = zip(white_pix_latest[0], white_pix_latest[1])
        pix=len(white_pix_ref_c)
        for m in white_pix_ref_c:
            for n in white_pix_latest_c:
                if m==n:
                    simpix=simpix+1
                    break
        percentsim= float(simpix)/pix*100
        return percentsim

    except:
        pix=len(white_pix_ref_c)
        simpix=0
        percentsim= float(simpix)/pix*100
        return percentsim

#Function to implement voting 
def prediction(decisionarray,mse_threshold,pixel_threshold,hist_threshold,line_treshold):
    decision=0
    if(decisionarray[0]> mse_threshold):
        decision=decision+1;
    if (decisionarray[1]> pixel_threshold or decisionarray[1]< -(pixel_threshold)):
        decision=decision+1;
    if(decisionarray[2]< hist_threshold):
        decision=decision+1;
    if(decisionarray[3]< line_treshold):
        decision=decision+1;
    if (decisionarray[0]>11000):
        decision= -1;
    return decision

# Function to create mail body and send the mail with the predicted result
def mailcontent(wdir,to_recipients,result,cam,msgdatetodel):
    result=result
    print result
    printresult=''
    foldernames=os.listdir(wdir)
    imgHtml=''

    if(cam in foldernames):
        camname=cam;
        #print camname            
        filenames=os.listdir(os.path.join(wdir,cam))
        filearray=np.asarray(filenames)
        filearraysorted= np.sort(filearray)
        n= filearraysorted.size
        try:
            file_date_back=daystofetch/7
            filename1= os.path.join(wdir,cam,filearraysorted[n-(file_date_back)])
            filename2=os.path.join(wdir,cam,filearraysorted[n-1])            
            # Declare D as date from file
            D1= ('20'+filearraysorted[n-(file_date_back)].split(camname,1)[1][0:12])
            D2= ('20'+filearraysorted[n-1].split(camname,1)[1][0:12])
            date1 = datetime(year=int(D1[0:4]), month=int(D1[4:6]), day=int(D1[6:8]),hour=int(D1[8:10]), minute=int(D1[10:12]),second=int(D1[12:14]))
            datelastmonth=date1.strftime("%B %d, %Y %H:%M %p")
            date2 = datetime(year=int(D2[0:4]), month=int(D2[4:6]), day=int(D2[6:8]),hour=int(D2[8:10]), minute=int(D2[10:12]),second=int(D2[12:14]))
            daterecent=date2.strftime("%B %d, %Y %H:%M %p")
                           
            if (result==2 or result==3):
                printresult="The " +refdetails[i].split('_')[0] +" well might be Dirty"
            elif(result>3):
                printresult= "The " +refdetails[i].split('_')[0] +" well should be Cleaned"
            elif (result==-1):
                printresult= "The camera " +refdetails[i].split('_')[0] +" might be faulty "
            else: 
                printresult= "The " +refdetails[i].split('_')[0] +" well is Clean "
            with open(filename1, 'rb') as f:
            
                my_logo1 = FileAttachment(name=filename1, content=f.read(), is_inline=True, content_id=filename1)
                m.attach (my_logo1)
            f.close()
        
            with open(filename2, 'rb') as f:
                my_logo2 = FileAttachment(name=filename2, content=f.read(), is_inline=True, content_id=filename2)
                m.attach(my_logo2)
            f.close()    
            if (result==2 or result==3):
                imgHtml +="""<hr><br> 
                <span style="background-color: #FFFF00">
                <h2 align="center">Camera Location: %s - may require cleaning </h2>
                </span>
        <table >
        <th>Last Month (%s) </th><th>Most Recent (%s)</th>
        <tr>
        <td><img src="cid:%s" width="500" height="500"></td>
        <td><img src="cid:%s" width="500" height="500"></td>
        </tr>
        <span style="color: #FF0000">
        <tr style="font-weight:bold">
        <td>%s</td>
        <td align="center" font size="10"> %s</td>
        </tr>
        </span>
        </table>
        </br>""" %(camname,datelastmonth,daterecent,filename1,filename2,'',printresult)
            elif(result>3):
                imgHtml +="""<hr><br> 
                <span style="background-color: #FF0000">
                <h2 align="center">Camera Location: %s - Should be cleaned </h2>
                </span>
        <table >
        <th>Last Month (%s) </th><th>Most Recent (%s)</th>
        <tr>
        <td><img src="cid:%s" width="500" height="500"></td>
        <td><img src="cid:%s" width="500" height="500"></td>
        </tr>
        <span style="color: #FF0000">
        <tr style="font-weight:bold">
        <td>%s</td>
        <td align="center" font size="10"> %s</td>
        </tr>
        </span>
        </table>
        </br>""" %(camname,datelastmonth,daterecent,filename1,filename2,'',printresult)

            elif(result==-1):
                imgHtml +="""<hr><br> 
                <span style="background-color: #FFFF00">
                <h2 align="center">Camera Location: %s - may be faulty </h2>
                </span>
        <table >
        <th>Last Month (%s) </th><th>Most Recent (%s)</th>
        <tr>
        <td><img src="cid:%s" width="500" height="500"></td>
        <td><img src="cid:%s" width="500" height="500"></td>
        </tr>
        <span style="color: #FF0000">
        <tr style="font-weight:bold">
        <td>%s</td>
        <td align="center" font size="10"> %s</td>
        </tr>
        </span>
        </table> 
        </br>""" %(camname,datelastmonth,daterecent,filename1,filename2,'',printresult)
            
            else:
               
                imgHtml += """<hr><br> 
                <h2 align="center">Camera Location: %s  </h2>
            <table > <th>Last Month (%s) </th><th>Most Recent (%s)</th>
            <tr>
            <td><img src="cid:%s" width="500" height="500"></td>
            <td><img src="cid:%s" width="500" height="500"></td>
            </tr>
            <tr style="font-weight:bold">
            <td>%s</td>
            <td align="center" font size="10"> %s</td>
            </tr>
            </table>
            </br>""" %(camname,datelastmonth,daterecent,filename1,filename2,'',printresult)
            return imgHtml
        except:
            print "No files found for camera for "+camname+ " which is in the folder structure "+ wdir+" or look \
            whether files has got similar formats";                
            sleep(2)                              
            sys.exit()  
                
  
tz = EWSTimeZone.localzone()
right_now = tz.localize(EWSDateTime.now())

#Getting required values from config file
daystofetch=int(config.get('configurations','daystofetch'))
maildays=int(config.get('configurations','maildays'))
maildays= fetchdate(right_now,maildays)
msgdatetodel= fetchdate(right_now,daystofetch)
img_ref_path=config.get('configurations','img_ref') 
wdir=config.get('configurations','wdir')
dirnames=config.get('configurations','dirnames').split(",")
createfolder(wdir,dirnames)
fetchmailattachments(maildays,right_now,wdir)
camname=[]
refdetails=[]
decisionarray=[]
imgHtml=''
for each_section in config.sections():
    if each_section != 'configurations':
        for (each_key, each_val) in config.items(each_section):
            camname.append(each_section)
            refdetails.append(each_val)

for i in range (0,len(refdetails),7):
    decisionarray=[]
    image_ref_cut_grey, image_ref_cut_colour= getRefImages(img_ref_path,refdetails[i],refdetails,0)
    image_ref_grey_line,image_ref_colour_line=getRefImages(img_ref_path,refdetails[i+1],refdetails,1)
    image_latest_cut_grey, image_latest_cut_colour= getLatestImages(wdir,refdetails[i].split('_')[0],refdetails,0)
    image_latest_grey_line, image_latest_colour_line= getLatestImages(wdir,refdetails[i].split('_')[0],refdetails,1)
    decision1= meansquare(image_ref_cut_grey,image_latest_cut_grey)
    decisionarray.append(decision1)
    decision2=pixeldirt(image_ref_cut_grey,image_latest_cut_grey)
    decisionarray.append(decision2)
    decision3= histogram (image_ref_cut_colour,image_latest_cut_colour)
    decisionarray.append(decision3)
    decision4=edgedetection(image_ref_grey_line,image_latest_grey_line)
    decisionarray.append(decision4)
    result= prediction(decisionarray,float(refdetails[i+3]),float(refdetails[i+4]),float(refdetails[i+5]),float(refdetails[i+6]))
    imgHtml+=str(mailcontent(wdir,to_recipients,result,refdetails[i].split('_')[0],daystofetch))
# orb algorithm not used now. Can be used to find whether a camera is faulty if the function return 0(no matches)
    #decision2= orbalgorithm(image_ref_cut_grey,image_latest_cut_grey)
    #decisionarray.append(decision2)
m.body = HTMLBody(imgHtml)
try:
    m.send()
    img_del = account.inbox.filter(datetime_received__range=(msgdatetodel, right_now))
    for item in img_del:
        item.move_to_trash()
    print "Message Sent Well Done!!"
        
except:
    print "Message Not Sent"

    
    
        

