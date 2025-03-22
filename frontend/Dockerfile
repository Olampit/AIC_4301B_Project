# Utilisation d'une image de node légère
FROM node:18-alpine

# Répertoire de travail
WORKDIR /app

# Copier tous les fichiers frontend dans le répertoire de travail
COPY . .

# Installer 'serve' globalement
RUN npm install -g serve

# Exposer le port 3000
EXPOSE 3000

# Démarrer le serveur statique avec 'serve'
CMD ["serve", "-s", ".", "-l", "3000"]
