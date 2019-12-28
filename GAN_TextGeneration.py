from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
#import keras.utils as ku 
import numpy as np
from keras.utils import np_utils

from keras.optimizers import Adam

from tqdm import tqdm

def load_data(filename):
    #load data
    data = open(filename).read()
    
    #seperate data sentence by sentence
    sentences = data.split(".")
    
    #list unique character in the data and sort them
    letter = list(sorted(set(data)))
    
    #tokenize every character by dictionary

    #every letter is key and number is value
    mapping = dict((char,index) for index, char in enumerate(letter))
    
    #every number is key and letter is value
    inversMap = dict((index,char) for index, char in enumerate(letter))
    
    #find the longest sentence in the data later we train data compare by this size
    longestSentence = max([len(sentence) for sentence in sentences])
    
    character_sequence = []
    
    #turn every character into integer (tokenize) based on map(dictionary) so that 
    #we can use them on networ
    for char in data:
        for charvalue,value in mapping.items():
            if(char == charvalue):
                character_sequence.append(value)
    
    values = []
    character_sentences = []
    #seperate every sentence in tokenize data and addd them in to character sentence
    for value in character_sequence:
        values.append(value)
        if(value == mapping["."]):
            character_sentences.append(values)
            values = []
    
    #create input data: pad every list in character_sentences fix size(longestSentence)
    #then turn it to numpy array so that we can use it to network
    input_sequences = np.array(pad_sequences(character_sentences, maxlen=longestSentence, padding='pre'))

    return letter,mapping,inversMap,longestSentence,input_sequences

def create_generator(longestSentence,vocabsize):
    
    generator = Sequential()
    generator.add(LSTM(256, input_shape=(vocabsize, 1),return_sequences=True))
    generator.add(Dropout(0.2))
    generator.add(LSTM(vocabsize,return_sequences=True))
    generator.add(Dropout(0.2))
    #we use LSTM for last layer  instead of Dense layer
    #so that we can create discriminator start with LSTM layer when we combine 
    #them for create GAN
    generator.add(LSTM(1,return_sequences=True,activation='softmax'))
    
    #################---Extras---###########################
    #we compile generator just only pre train before add it into gan
    generator.compile(loss='binary_crossentropy', optimizer='adam')
    ########################################################
    
    generator.summary()
    return generator
        
def create_discriminator(longestSentence,vocabsize):
    
    discriminator = Sequential()
    discriminator.add(LSTM(1,input_shape=(vocabsize,1),return_sequences=True))
    discriminator.add(Dropout(0.4))
    discriminator.add(LSTM(50))
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units=1,activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy',optimizer=opt)
    
    discriminator.summary()
    return discriminator
    
def create_gan(generator,discriminator):

    #connect them
    gan = Sequential()
    #add generator
    gan.add(generator)
    #add the discriminator
    gan.add(discriminator)
    #compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=opt)
    
    gan.summary()
    
    return gan
    
def generate_text(generator,inversMap,input_sequences,nextchar):
    
    #create noise vector (not by letter, by value of letter)
    randomStart = np.random.randint(0, len(letter),size=len(letter)) 
    
    for i in tqdm(range(len(letter))):
        #reshape noise vector into 3 dimension so that it can be used with LSTM layer
        x = np.reshape(randomStart, (1, len(randomStart), 1))
        #normalize x
        x = x/float(len(mapping))
        #predict a letter
        pred = generator.predict(x)
        index = np.argmax(pred)
        #add a value of letter in our output array
        randomStart = np.append(randomStart, index)
        randomStart = randomStart[1: len(randomStart)]

    #turn output array(value of letters) into string with letter
    generated = "".join([inversMap[value] for value in randomStart])
    return generated

#this function takes a character string and turn it into value of letter array according to map
def character_tokenize(sample,mapping):
    
    character_sequence = []
    
    for char in sample:
        for charvalue,value in mapping.items():
            if(char == charvalue):
                character_sequence.append(value)
    sample_token = np.asarray(character_sequence )           
    return sample_token

#this function takes value of letter arrays and turn it into the string according to inversMap
def token_to_character(sample,inversMap):
    text = ""
    for index in sample:
        for value,char in inversMap.items():
            if(index == value):
                text += char
    return text

#################---Extras---###########################
#train generator before add it to GAN ,
def train_Generator(generator,input_sequences,letter,epoch):
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=len(letter)+2, padding='pre'))
    
    X = input_sequences[:,:-2]
    X = np.reshape(X, (len(X), len(letter), 1))
    label = input_sequences[:,-2]
    
    label = np.reshape(label, (len(label), 1, 1))
    label = np_utils.to_categorical(label)
    label = np.reshape(label, (len(label), len(letter), 1))
    
    generator.fit(X, label, epochs=epoch)    
########################################################

if __name__ == '__main__':
    
    #load the real and train data with character token map and inputs
    letter,mapping,inversMap,longestSentence,input_sequences = load_data('training.txt')
    
    #load the Invalid and train data with character token map and inputs
    inLetter,inMapping,inInversMap,inLongestSentence,inInput_sequences = load_data('invalid.txt')#in=invalid
    
    #pad the real and valid input data to fix size
    inInput_sequences = np.array(pad_sequences(inInput_sequences, maxlen=len(letter), padding='pre'))
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=len(letter), padding='pre'))
    
    #create generator,discriminator model
    generator = create_generator(longestSentence,len(letter))
    discriminator = create_discriminator(longestSentence,len(letter))
    
    #################---Extras---###########################
    """
    #pretrain generator
    train_Generator(generator,input_sequences,letter,100)
    I am not sure it works fine,first train gan with pretrained generator
    """
    ########################################################
    
    #crate gan model
    gan = create_gan(generator,discriminator)
  
    #start training
    epochs = 100 
    sample = 5
    for e in range(epochs):
        
        print("*********** Epoch",e+1,"***********")
        
        #make discriminator trainable true so that we can train it
        discriminator.trainable = True
        
        #create real sample vector
        real_samples = []
        for _ in range(sample):#create 5 real sample
            #real_samples.append(input_sequences[np.random.randint(0, len(input_sequences)-1)])
            real_txt = input_sequences[np.random.randint(0, len(input_sequences)-1)]
            x = np.reshape(real_txt, (len(real_txt), 1)) #reshape vector into 3 dimension so that we can use it by LSTM
            x = x/float(len(mapping))
            real_samples.append(x)
        
        #create fake sample vector
        fake_samples = []
        for _ in range(int(sample/sample)):#create 1 fake sample
            generated_text = generate_text(generator,inversMap,input_sequences,nextchar=longestSentence)
            generated_txt = character_tokenize(generated_text,mapping)
            x = np.reshape(generated_txt, (len(generated_txt), 1))#reshape vector into 3 dimension so that we can use it by LSTM
            x = x/float(len(mapping))
            fake_samples.append(x)
        
        labelValid = [1]*sample #label all real sample as 1
        labelFake = [0]*(int(sample/sample)) #label all fake sample as 0
        
        #concatenate labels , concatenate samples in true order
        label = np.concatenate([labelValid,labelFake])
        train = np.concatenate([real_samples,fake_samples])
        
        #train discriminator
        discriminator.train_on_batch(train,label)
        
        #after train discriminator  turn discriminator trainable false since 
        #we are going to train adversarial model
        discriminator.trainable = False
        
        #create noise sample vector
        noise_sample = []
        for _ in range(sample):
            noise_list = np.random.randint(0, len(letter),size=len(letter))
            x = np.reshape(noise_list, (len(noise_list), 1))#reshape vector into 3 dimension so that we can use it by LSTM
            x = x/float(len(mapping))
            noise_sample.append(x)
        
        #label all noise vector as 1 in order to fool the discriminator
        noise_label = [1]*sample
        noise_sample = np.asarray(noise_sample)

        #train generative model    
        gan.train_on_batch(noise_sample,noise_label)
        
        #print the generating sample and real sample
        print("*********************")
        print("generated txt:",token_to_character(generated_txt,inversMap))
        print("*********************")
        print("real txt:",token_to_character(real_txt,inversMap))
        print("*********************")
    
    
    #our final part we take vectors from valid datasets and invalid datasets to
    #test how our discriminators work well,evaluate it and print the loss of discriminator 
    
    sample = 100    
    valid_samples = []
    for _ in range(sample):
        valid = input_sequences[np.random.randint(0, len(input_sequences)-1)]
        x = np.reshape(valid, (len(valid), 1))
        x = x/float(len(mapping))
        valid_samples.append(x)
    
    invalid_samples = []
    for _ in range(sample):
        invalid = inInput_sequences[np.random.randint(0, len(inInput_sequences)-1)]
        x = np.reshape(invalid, (len(invalid), 1))
        x = x/float(len(mapping))
        invalid_samples.append(x)
    
    labelValid = [1]*sample
    labelInvalid = [0]*sample
    
    labelTest = np.concatenate([labelValid,labelInvalid])
    test = np.concatenate([valid_samples,invalid_samples])
    
    loss = discriminator.evaluate(x=test, y=labelTest)
    
    print("Loss Value of discriminator after training:",loss)
