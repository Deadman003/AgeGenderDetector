import cv2 

faceproto = "AgeGenderDetector/Models/opencv_face_detector.pbtxt"
facemodel = "AgeGenderDetector/Models/opencv_face_detector_uint8.pb"
ageproto = "AgeGenderDetector/Models/age_deploy.prototxt"
agemodel = "AgeGenderDetector/Models/age_net.caffemodel"
genproto = "AgeGenderDetector/Models/gender_deploy.prototxt"
genmodel = "AgeGenderDetector/Models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) 

faceNet = cv2.dnn.readNet(facemodel, faceproto) 
ageNet = cv2.dnn.readNet(agemodel, ageproto) 
genNet = cv2.dnn.readNet(genmodel, genproto) 

agelist = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
genlist = ['Male', 'Female'] 


def faceBox(faceNet, frame) :
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [106.13,115.97,124.96], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),1)
    return frame, bbox


video = cv2.VideoCapture(0)

while True:
    ret,frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
    for bbox in bboxs:
        face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (277,277), MODEL_MEAN_VALUES, swapRB=False)

        genNet.setInput(blob)
        genpred = genNet.forward()
        gender = genlist[genpred[0].argmax()]

        ageNet.setInput(blob)
        agepred = ageNet.forward()
        age = agelist[agepred[0].argmax()]

        label = "{},{}".format(gender,age)
        cv2.putText(frame, label, (bbox[0],bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender", frame)
    if  cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()