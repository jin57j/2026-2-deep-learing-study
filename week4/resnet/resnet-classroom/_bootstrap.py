"""ResNet 단독 실행에서도 week4/shared를 찾을 수 있도록 경로를 추가한다."""

from pathlib import Path
import sys


week4_path = str(Path(__file__).resolve().parents[2])
if week4_path not in sys.path:
    sys.path.insert(0, week4_path)
