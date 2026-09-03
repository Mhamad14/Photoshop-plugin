@echo off
echo ======================================================================
echo YOLO TRAINING - 60 Epochs on ACNE04 Dataset
echo ======================================================================
echo This will take 4-6 hours on CPU. Keep this window open.
echo.
echo Training logs: backend/training/training_log.txt
echo Model output: backend/training/runs/segment/runs/acne04_60ep/weights/best.pt
echo.
echo Starting training...
echo.

cd /d "%~dp0"
py -3.13 train_yolo_seg.py --data dataset.yaml --epochs 60 --imgsz 640 --batch 16 --device cpu --name acne04_60ep 2>&1 | tee training_log.txt

echo.
echo ======================================================================
echo Training complete! Check runs/segment/runs/acne04_60ep/weights/best.pt
echo ======================================================================
pause
