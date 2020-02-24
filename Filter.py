import cv2

cap = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')
mustache = cv2.imread("mustache.png",-1)
glasses = cv2.imread("glasses.png",-1)

while True:
    ret,img = cap.read()
    if ret == False:
        continue
    
    eyes = eye_cascade.detectMultiScale(img,1.1,5)
    nose = nose_cascade.detectMultiScale(img,1.1,5)
    for (x,y,w,h) in eyes:
        x1 = x-10
        y1 = y-15
        x2 = x+h+10
        y2 = y+w+w//4
        glasses = cv2.resize(glasses,(x2-x1,y2-y1))
        alpha_mask = glasses[:,:,3]/255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(3):
            img[y1:y2,x1:x2,c] = alpha_mask * glasses[:,:,c] + alpha_inv * img[y1:y2,x1:x2,c]
    
    for (x,y,w,h) in nose:
        x1 = x-w//4
        y1 = y+h//3+w//7
        x2 = x+h+h//2
        y2 = y+w+h//2
        mustache = cv2.resize(mustache,(x2-x1,y2-y1))
        alpha_mask = mustache[:,:,3]/255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(3):
            img[y1:y2,x1:x2,c] = alpha_mask * mustache[:,:,c] + alpha_inv * img[y1:y2,x1:x2,c]
    
    cv2.imshow("Video Frame",img)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
