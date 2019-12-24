# projet_cnn

## Ce qui marche

Le modèle utilisé est composé d'une couche de convolution à 10 filtres, d'un maxpooling, et d'une couche dense de 10 neurones. 

La fonction de coût utilisée est categorical-crossentropy, qui applique la fonction softmax à l'output (raison pour laquelle il n'y a pas de fonction d'activation à la dernière couche).

En ne faisant apprendre que la dernière couche, le modèle s'améliore : accuracy de 0.89 sur les 1000 premières images de test, pour un entraînement sur les 5000 premières images de train.

## Ce qui ne marche pas

Quand on enchaine deux couches denses dans le modèle, la deuxième couche finit par renvoyer des valeurs beaucoup trop grandes : je pense donc m'être trompé sur la valeur de retour de la backprop. 

J'ai essayé de corriger cette erreur, mais je retombais tout le temps sur la même formule en faisant la dérivée.

J'ai quand même implémenté la backprop pour les autres couches (maxpool et conv2D), mais sans modifier les poids, pour ne pas avoir le comportement décrit précedemment. Je n'ai donc pas pu tester l'apprentissage des poids des filtres. 

Aussi, la fonction de backpropagation de Conv2D (renommée backprop2 pour ne pas l'exécuter pendant la phase d'apprentissage) ne renvoie pas la bonne valeur, faute de temps. Ainsi, même si le problème précedemment décrit était résolu, il ne serait pour le moment pas possible d'enchainer les couches de convolution.

## Ce que je n'ai pas fait

L'apprentissage de fait pour le moment de manière sotchastique, je n'ai pas implémenté de méthode avec batch.

## Temps d'exécution

Le code produit étant très mal optimisé (pas assez de numpy, trop d'itérations inutiles), l'apprentissage est très très lent.