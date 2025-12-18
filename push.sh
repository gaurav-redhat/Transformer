#!/bin/bash
cd /home/ggoswami/Project/transformer_problems/transformer-blog
rm -rf .git
git init
git remote add origin git@github.com:gaurav-redhat/Transformer.git
git add .
git commit -m "🚀 Complete Transformer Guide: 2017-2025 with 41 diagrams"
git branch -M main
git push -u origin main --force
echo "✅ Push complete!"

