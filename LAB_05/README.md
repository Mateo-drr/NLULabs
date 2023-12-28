----------------------------------
        S                                                
   _____|___                                              
  |         VP                                           
  |      ___|_________________                            
  |     |                     S                          
  |     |         ____________|________                   
  |     |        |                     VP                
  |     |        |             ________|____              
  |     |        |            |           PP-CLR         
  |     |        |            |     ________|_____        
NP-SBJ  |      NP-SBJ         |    |              NP     
  |     |    ____|______      |    |         _____|___    
  EX   VBZ  DT          NN   VBD   IN       DT        NN 
  |     |   |           |     |    |        |         |   
there   is  a         plane stuck  in       a        tree

None
             S                                                                            
   __________|__________                                                                   
  |                     VP                                                                
  |      _______________|__________________                                                
  |     |                                  VP                                             
  |     |     _____________________________|____                                           
  |     |    |                                  VP                                        
  |     |    |     _____________________________|___________                               
  |     |    |    |     |        |                        PP-MNR                          
  |     |    |    |     |        |               ___________|______                        
  |     |    |    |     |        |              |                S-NOM                    
  |     |    |    |     |        |              |     _____________|____                   
  |     |    |    |     |        |              |    |                  VP                
  |     |    |    |     |        |              |    |       ___________|___               
  |     |    |    |     |      PP-CLR           |    |      |               VP            
  |     |    |    |     |    ____|_____         |    |      |       ________|____          
NP-SBJ  |    |    |     NP  |          NP       |  NP-SBJ   |      |             NP       
  |     |    |    |     |   |     _____|___     |    |      |      |         ____|____     
 PRP   VBP   TO   VB   PRP  TO   DT        JJ   IN  PRP     MD    VBD      PRP$      NNS  
  |     |    |    |     |   |    |         |    |    |      |      |        |         |    
  I    want  to return  it  to  the       wild  so   it    can    lay      its      things

None
            S                         
   _________|_____                     
  |               VP                  
  |      _________|_____               
  |     |   |           VP            
  |     |   |      _____|___           
NP-SBJ  |   |     |         NP        
  |     |   |     |      ___|_____     
 PRP    MD  RB    VB    DT        NN  
  |     |   |     |     |         |    
  I    can not control the     weather

None
------------------cstm-----------------
                S                         
        ________|______________            
       |                       VP         
       |               ________|___        
       NP             |            PP     
   ____|________      |     _______|___    
  DT   V   DT   N     V    P       DT  N  
  |    |   |    |     |    |       |   |   
there  is  a  plane stuck  in      a  tree

None
                               S                                    
                 ______________|__________________                   
                NP                                |                 
  ______________|______________                   |                  
 |         VP                  PP               S-NOM               
 |     ____|__________      ___|___      _________|_____________     
PRON  V    TO   V    PRON  TO  DT  N    IN PRON   MD   V  PRP   N   
 |    |    |    |     |    |   |   |    |   |     |    |   |    |    
 I   want  to return  it   to the wild  so  it   can  lay its things

None
          S                     
  ________|_____                 
 NP             VP              
 |     _________|___________     
PRON  MD  RB    V     DT    N   
 |    |   |     |     |     |    
 I   can not control the weather

None

Generated sentences using nltk.parse.generate:
['there', 'is', 'there', 'plane', 'is', 'in', 'there', 'plane']
['there', 'is', 'there', 'plane', 'is', 'in', 'there', 'tree']
['there', 'is', 'there', 'plane', 'is', 'in', 'there', 'wild']

Generated sentences using PCFG:
the stuck there wild can not return a wild
I can not is the wild in there tree control to a tree
it return to the weather
I stuck to a weather in there plane lay to want I
there return the plane control to the things
it return to control it
the stuck a plane want to a tree
it is in a things to a things can not is a plane
it control to a things
I want to the tree in a tree can not lay a tree