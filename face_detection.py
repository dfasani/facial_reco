#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import traceback
import cv2
import datetime
import io
import os
from google.cloud import vision


class FaceDetection:
    def __init__(self, image_path, archive_folder="/tmp/", debug=False):
        """ init
            @image_path : le chemin d'une image sur le disque
            @archive_folder : dossier d'archive
            @debug : si True, affichage des images dans une fenêtre
        """
        logging.info("Image : {0}".format(image_path))
        self.image_path = image_path
        self.archive_folder = archive_folder
        self.debug = debug
        self.items = []
        self.items_frames = []

        # Afin de grouper les archives, nous allons utiliser un préfixe unique
        self.images_prefix = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))

        # On charge l'image dans une frame 
        self.frame = cv2.imread(image_path)

        # Affichage de l'image originale
        if self.debug:
            cv2.imshow("preview", self.frame)
            cv2.waitKey()

    def find_items(self):
        """ Trouver les items dans une frame

            Valorise self.items en tant que liste contenant les coordonnées des viages au format (x, y, h, w).
            Exemple : 
                      [[ 483  137   47   47]
                       [ 357  152   46   46]
                       ...
                       [ 126  167   51   51]]

        """
        logging.info("Recherche des items...")

        items = []

        vision_client = vision.Client()

        with io.open(self.image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision_client.image(content=content)

        faces = image.detect_faces()

        for face in faces:
            x1 = face.bounds.vertices[0].x_coordinate
            y1 = face.bounds.vertices[0].y_coordinate

            x2 = face.bounds.vertices[2].x_coordinate
            y2 = face.bounds.vertices[2].y_coordinate

            items.append((x1, y1, x2 - x1, y2 - y1))

        # On valorise self.items et on affiche un peu de log
        logging.info("Nombre d'items : '{0}'".format(len(items)))
        logging.info("Items = {0}".format(items))
        self.items = items

    def extract_items_frames(self):
        """ Extraire les frames des items de la frame complète
            Valorise self.items_frames en tant que liste des frames et coordonnées.
            Exemple : 
                      [
                        {  "frame" : ...,
                           "x" : ...,
                           "x" : ...,
                           "x" : ...,
                           "x" : ...
                        },
                        { ... },
                        ...
                      ]
        """
        logging.info("Extractions des frames des items ('{0}' à extraire)...".format(len(self.items)))
        items_frames = []
        # pour chaque coordonnées d'items...
        for f in self.items:
            # On extrait le sous ensemble de la frame complète
            x, y, w, h = f
            item_frame = self.frame[y:y + h, x:x + w]
            # Et on le stocke ainsi que ses coordonnées
            items_frames.append({
                "frame": item_frame,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })

            # On affiche chaque visage extrait dans une fenêtre
            # if self.debug:
            #    cv2.imshow("preview", item_frame)
            #    cv2.waitKey()

        self.items_frames = items_frames

    def get_items_frames(self, grayscale=False):
        """ Retourne les frames des items et leurs coordonnées dans une liste
            @grayscale : si True, retourne les frames des items en niveaux de gris
        """
        # Si on ne désire pas un retour en niveau de gris, on retourne les données telles quelles
        if not grayscale:
            return self.items_frames

        # Dans le cas contraire, on créée une liste temporaire et pour chaque frame de la liste 
        # originale, on insère dans la nouvelle liste la frame convertie en niveaux de gris
        items_frames = []
        for item_frame in self.items_frames:
            item_frame["frame"] = cv2.cvtColor(item_frame["frame"], cv2.COLOR_BGR2GRAY)
            items_frames.append(item_frame)
        return items_frames

    def add_label(self, text, x, y):
        """ Ajout d'un label sur la frame complète
            @text : texte à afficher
            @x, y : coordonnées du texte à afficher
        """
        # Comme cette fonction sera utilisée pour afficher un label sur un carré autour d'un visage,
        # on ajoute volontairement un peu d'espace entre le label et le carré, sauf si on est au bord de l'image 
        if y > 11:
            y = y - 5
        cv2.putText(self.frame, text, (x+40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

    def archive_items_frames(self):
        """ Ecrit dans le dossier d'archive chaque frame de chaque item en tant qu'une image
        """
        logging.info("Archive les items ('{0}' à archiver)...".format(len(self.items_frames)))
        idx = 0
        # Pour chaque item, on le sauve dans un fichier 
        for item_frame in self.items_frames:
            a_frame = item_frame["frame"]
            image_name = "{0}_item_{1}.jpg".format(self.images_prefix, idx)
            logging.info("Archive un item dans le fichier : '{0}'".format(image_name))
            cv2.imwrite(os.path.join(self.archive_folder, image_name), a_frame)
            idx += 1

    def archive_with_items(self):
        """ Ecrit dans le dossier d'archive la frame complète avec des carrés dessinés autour
            des visages détectés
        """
        logging.info("Archive l'image avec les items trouvés...")
        # Dessine un carré autour de chaque item
        for f in self.items:
            x, y, w, h = f  # [ v for v in f ]
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Ajoute la date et l'heure à l'image
        cv2.putText(self.frame, datetime.datetime.now().strftime("%c"), (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

        # On affiche l'image qui va être archivée dans une fenêtre
        if self.debug:
            cv2.imshow("preview", self.frame)
            cv2.waitKey()

            # Ecriture du fichier
        archive_full_name = "{0}_full.jpg".format(self.images_prefix)
        logging.info("Archive file is : '{0}'".format(archive_full_name))
        cv2.imwrite(os.path.join(self.archive_folder, archive_full_name), self.frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Test...")
    T = FaceDetection("./test/test (1).jpg",
                      archive_folder="./archives/",
                      debug=False)
    T.find_items()
    T.extract_items_frames()


    for item in T.get_items_frames(grayscale=True):
        label = "coucou"
        logging.info("Trouvé : {0}".format(label))
        x = item["x"]
        y = item["y"]
        T.add_label(label, x, y)

    T.archive_items_frames()
    T.archive_with_items()
