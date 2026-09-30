import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

corpus = [
    ("God and faith discussion regarding secular humanism", "alt.atheism"),
    ("Rendering 3D models with ray tracing software", "comp.graphics"),
    ("Windows system drivers and operating system installation", "comp.os.ms-windows.misc"),
    ("PCI bus controller motherboard memory configuration", "comp.sys.ibm.pc.hardware"),
    ("Apple Macintosh monitor and keyboard port issue", "comp.sys.mac.hardware"),
    ("X11 server window manager configuration Unix", "comp.windows.x"),
    ("For sale brand new digital camera cheap price", "misc.forsale"),
    ("Automobile engine transmission and brake pads oil", "rec.autos"),
    ("Motorcycle helmet riding Yamaha speed bike", "rec.motorcycles"),
    ("Baseball stadium pitcher home run batting average", "rec.sport.baseball"),
    ("Hockey playoffs Stanley cup goalkeeper puck shot", "rec.sport.hockey"),
    ("Encryption RSA algorithm public key security privacy", "sci.crypt"),
    ("Circuit board resistors capacitors soldering voltage", "sci.electronics"),
    ("Medical diagnosis treatment patients disease symptoms", "sci.med"),
    ("NASA satellite rocket telescope orbit space exploration", "sci.space"),
    ("Church pastor gospel biblical teachings Jesus Christ", "soc.religion.christian"),
    ("Firearms gun control rights amendment weapons", "talk.politics.guns"),
    ("Peace talks Middle East diplomatic negotiations territory", "talk.politics.mideast"),
    ("Government elections voting parliament political parties", "talk.politics.misc"),
    ("Religious beliefs theology spiritual morality values", "talk.religion.misc")
]

texts, labels = zip(*corpus)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

test_doc = ["Doctor diagnosed patient with medical symptoms and treatment medicine"]
prediction = model.predict(test_doc)[0]

print("Input Text:", test_doc[0])
print("Predicted Newsgroup Category:", prediction)
