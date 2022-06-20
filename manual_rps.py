'''
import cv2


from keras.models import load_model 
import numpy as np 

model = load_model('keras_model.h5')
capture = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True: 
    ret, frame = capture.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalise the image 
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()

cv2.destroyAllWindows() '''

import random 
def play():
    user = input("What's yout choice? 'r' for rock, 'p' for paper, 's' for scissors\n")
    user = user.lower()

    computer = random.choice(['r', 'p', 's'])
    print(computer)

    if user == computer: 
        return "You and the computer have both chosen {}. It's a tie!".format(computer)

    if is_win(user, computer):
        return "you have chosen {} and the computer has chosen {}. YOU WON!".format(user, computer)
    
    return "You have chosen {} and the computer has chosen {}. You lost :(".format(user, computer)

def is_win(player, opponent):
    # return true if the player beats the opponent
    # winning conditions: r > s, s > p, p > r
    if (player == 'r' and opponent == 's') or (player == 's' and opponent == 'p') or (player == 'p' and opponent == 'r'):
        return True 
    return False 

if __name__ == '__main__':
    print(play())
