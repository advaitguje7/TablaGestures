# TablaGestures

A real-time computer vision system that allows a user to count through Indian classical rhythm cycles (taals) on their hand using thumb–finger gestures.

The system maps thumb pinches to discrete beats (matras), displays the corresponding bol, and supports gesture-based tala switching.

## Quickstart

**Requirements**
- Python 3.9+
- Webcam
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/tabla_gesture.py
```

# Tala Overview and Introduction of Terms
A **taal / tala** is a rhythm cycle in Indian Classical Music, at its simplest definition. Analogous to Western time signatures, a taal is a standardized division of beats that allows musicians to communicate ideas efficiently. This project applies the Hindustani (North Indian) Tala framework, which differs to the Carnatic (South Indian) framework in certain fundamental areas, namely, the concept of Theka.

A **Theka** is a fixed cycle of syllables in Hindustani music. For example, the 16 beat rhythm cycle known as 'TeenTaal' is as follows: <br>
Dha Dhin Dhin Dha | Dha Dhin Dhin Dha | Dha Tin Tin Ta | Ta Dhin Dhin Dha

In this project, the Thekas for 4 rhythm cycles have been implemented: Teentaal (16), Ektaal (12), Jhaptaal (10), and Rupak (7). An additional Carnatic talam, known as Adi Talam, is implemented as well. However, since the Carnatic system does not utilize the concept of Theka, a generally taught variation of AdiTalam is used.

Some other terms:
**Matra:** The beat number
**Bol:** The verbal syllable corresponding to the drum stroke
**Sam:** The first beat of a taal
**Khali:** Literally, the empty/open beat of the taal 

#  Understanding the Implementation

This project uses OpenCV to track 16 landmarks on the user's four fingers, utilizing the thumb as a selector. Placing the thumb on one of these landmarks will display the distance between the thumb and landmark (pinch_px), the taal, and the bol at that specific landmark. There are four general gestures: measure select, matra select, reset, change taal. To exit the program, hit Esc.

## Measure Select
Due to the complexity and anatomical issues of dealing with 16 landmarks at the same time, I split the landmarks into four, with four beats per finger. 

To select a finger, simply touch your thumb to the corresponding fingertip, marked by a circle.
<img width="480" height="520" alt="image" src="https://github.com/user-attachments/assets/6cc0fa3a-3754-43ad-8d30-14f57d43ed15" />
<img width="480" height="520" alt="image" src="https://github.com/user-attachments/assets/3b02abda-896c-4908-87f2-5f1571f3f4f7" />
<img width="480" height="520" alt="image" src="https://github.com/user-attachments/assets/3475f368-3ffb-43a9-b47c-f47a46db0c5b" />

## Matra Select
To select a matra, touch a landmark with the thumb. The corresponding bol will be displayed on the screen. 
<img width="480" height="520" alt="image" src="https://github.com/user-attachments/assets/dbc24ca7-86f4-4637-ad89-754d1d72fadf" />
<img width="480" height="520" alt="image" src="https://github.com/user-attachments/assets/d3ab9f73-7c0b-4dd0-833c-f437f469bd01" />


## Reset 
To go to the next finger or change taal, one will need to reset to the Measure Select condition. To reset, simply curl your fingers, such that the fingertip-landmark intersects with the finger-base landmark as shown: <br>
<img width="480" height="520" alt="image" src="https://github.com/user-attachments/assets/37490486-4eb4-4803-b6a2-ea61ccadb0e5" />
<img width="480" height="520" alt="image" src="https://github.com/user-attachments/assets/2692b3c0-3afd-463c-8467-d49220b7e9c0" />

## Change Taal
To change the tala, cross the index and middle finger such that the two fingertip landmarks intersect as shown below. 
<img width="480" height="520" alt="image" src="https://github.com/user-attachments/assets/fe509860-500d-46a9-9184-0714560dde57" /> <br>
The talas cycle through the following list:
Teentaal <br>
Ektaal <br>
Jhaptaal <br>
Rupak <br>
AdiTalam <br>

## Implementation Notes

- Hand landmarks are tracked using MediaPipe HandLandmarker via OpenCV.
- Pinch distances are normalized using a wrist–MCP reference to maintain scale invariance.
- Hysteresis (PINCH_ON / PINCH_OFF) and cooldown timers are used to prevent flicker and repeated triggers.
- Landmarks are grouped per finger to avoid unavoidable anatomical overlap.

