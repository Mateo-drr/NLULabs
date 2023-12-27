# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import torch

def runSvm(data,txtlbls,numsp):
    
    # Convert string labels to binary
    label_encoder = LabelEncoder()
    lbls = label_encoder.fit_transform(txtlbls)
    
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Initialize the SVM classifier
    svm_classifier = SVC(C=1, kernel='sigmoid')

    # Initialize k-fold cross-validation
    kf = StratifiedKFold(n_splits=numsp, shuffle=True, random_state=1)

    kfmetrics2=[]
    for fold,(tidx, vidx) in enumerate(kf.split(data, lbls)):
        print(f'Fold {fold + 1}')
        X_train, X_test = [data[i] for i in tidx], [data[i] for i in vidx]
        y_train, y_test = [lbls[i] for i in tidx], [lbls[i] for i in vidx]

        # Fit and transform data
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
        # Train SVM
        svm_classifier.fit(X_train_tfidf, y_train)
    
        # Predictions on the test set
        y_pred = svm_classifier.predict(X_test_tfidf)
    
        # Evaluate the model
        kfmetrics2.append({'accuracy': accuracy_score(y_test, y_pred),
                          'precision': precision_score(y_test, y_pred),
                          'recall': recall_score(y_test, y_pred),
                          'f1': f1_score(y_test, y_pred)})
    
    return kfmetrics2 
    
def trainLoop(train_dl,train_ds,model,optimizer,criterion,epoch,device):
    train_loss=0
    model.train()
    for _,data in tqdm(enumerate(train_dl), total=int(len(train_ds)/train_dl.batch_size)):
        lbl = data['lbl'].to(device)
        text = data['text_input_ids'].to(device)
        att = data['text_attention_mask'].to(device)
        
        optimizer.zero_grad()
        out = model(text)
        loss = criterion(out,lbl)
        loss.backward()
        optimizer.step()
        
        train_loss +=loss.item()

    print('E',epoch+1,'TL',train_loss/len(train_ds))
    #print(lbl[0].to('cpu').detach().numpy()[0],out.to('cpu').detach().numpy()[0][0])
    
def validLoop(valid_dl,valid_ds,model,criterion,epoch,device,kfmetrics):
    oglbl,predlbl=[],[]
    model.eval()
    valid_loss=0
    with torch.no_grad():
        for data in valid_dl:
            lbl = data['lbl'].to(device)
            text = data['text_input_ids'].to(device)
            att = data['text_attention_mask'].to(device)
            
            out = model(text)
            loss = criterion(out,lbl)
            
            valid_loss +=loss.item()
            
            oglbl.append(round(lbl[0].to('cpu').detach().numpy()[0]))
            predlbl.append(round(out.to('cpu').detach().numpy()[0][0]))
        
        kfmetrics.append({'loss': valid_loss/len(valid_ds),
                          'accuracy': accuracy_score(oglbl, predlbl),
                          'precision': precision_score(oglbl, predlbl),
                          'recall': recall_score(oglbl, predlbl),
                          'f1': f1_score(oglbl, predlbl)})

    print('E',epoch+1,'VL',valid_loss/len(valid_ds))
    #print(lbl[0].to('cpu').detach().numpy()[0],out.to('cpu').detach().numpy()[0][0])
    
    return kfmetrics

def printMetrics(kfmetrics, name):
    avg_metrics = {
        'accuracy': np.mean([fold['accuracy'] for fold in kfmetrics]),
        'precision': np.mean([fold['precision'] for fold in kfmetrics]),
        'recall': np.mean([fold['recall'] for fold in kfmetrics]),
        'f1': np.mean([fold['f1'] for fold in kfmetrics])
    }
    
    print(f"Average Metrics {name}:")
    print(avg_metrics)
    return avg_metrics
    
def removeObjSents(dl,model,docsents,device,i,fsents,window):
    model.eval()
    with torch.no_grad():
        ftemp=[]
        for j,data in enumerate(dl):
            #check if sentence was broken
            if type(data[0]) is list:
                totpred = []
                for chunk in data:
                    #print('Breaking sentence', len(data), docsents[i][j])
                    chunk = chunk[0]['input_ids'].flatten().unsqueeze(0).to(device)
                    output = model(chunk) #predict if its objective or not
                    totpred.append(output.to('cpu').detach().numpy()[0][0])
                prediction = np.mean(totpred)
            else:
                data = data[0]['input_ids'].flatten().unsqueeze(0).to(device)
                prediction = model(data) #predict if its objective or not
                prediction = prediction.to('cpu').detach().numpy()[0][0]
                
            if round(prediction) <= window: #subjective, allow sentences that fall between +-0.1 of 0.5
               ftemp.append(docsents[i][j]) 
        fsents.append(ftemp) #build again the document without the objective sentences
    return fsents