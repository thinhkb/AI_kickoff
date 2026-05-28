"""
Batch extract all competition PDFs using MinerU (magic-pdf) CLI.
Automatically organizes the output .md files into data/processed/mineru_markdown/.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import PDF_DIR, MINERU_MD_DIR, ensure_dirs
from src.utils.logging_utils import logger
# Cấu hình: Đặt thành True nếu muốn kích hoạt trích xuất công thức toán học (MFD + MFR)
# Lưu ý: Khi bật, hệ thống sẽ tải thêm mô hình UniMERNet (~773MB) từ ModelScope và chạy chậm hơn trên CPU.
ENABLE_FORMULA = True



def check_mineru_cli() -> bool:
    """Check if mineru CLI is available in the current environment."""
    if shutil.which("mineru") is not None:
        return True
    # Fallback to sys.executable directory
    exe_dir = Path(sys.executable).parent
    fallback_paths = [exe_dir / "mineru.exe", exe_dir / "mineru"]
    for path in fallback_paths:
        if path.exists():
            return True
    return False


def extract_single_pdf(pdf_path: Path, temp_out_dir: Path) -> Optional[Path]:
    """
    Run mineru CLI on a single PDF file and return the path to the generated markdown.
    """
    pdf_stem = pdf_path.stem
    try:
        mineru_bin = "mineru"
        if shutil.which("mineru") is None:
            exe_dir = Path(sys.executable).parent
            fallback_paths = [exe_dir / "mineru.exe", exe_dir / "mineru"]
            for path in fallback_paths:
                if path.exists():
                    mineru_bin = str(path)
                    break

        # Diệt các tiến trình fast_api bị treo của MinerU để tránh ngốn GPU VRAM và giải phóng pipe trên Windows
        subprocess.run("taskkill /F /FI \"COMMANDLINE eq *mineru.cli.fast_api*\" /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Đọc số trang của tệp PDF để tính toán timeout động
        page_count = 1
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            page_count = len(reader.pages)
        except Exception as pe:
            logger.warning(f"Không thể đọc số trang của {pdf_path.name}: {pe}")
            
        # Tính timeout động: tối thiểu 5 phút (300s), hoặc 30 giây cho mỗi trang
        dynamic_timeout = max(300, page_count * 30)
        tqdm.write(f"-> Đang trích xuất {pdf_path.name} ({page_count} trang) - Timeout động: {dynamic_timeout} giây...")

        # Lệnh tiêu chuẩn: mineru -p <pdf_path> -o <temp_out_dir> -b pipeline -f <true/false>
        cmd = [
            mineru_bin,
            "-p", str(pdf_path),
            "-o", str(temp_out_dir),
            "-b", "pipeline",
            "-f", "true" if ENABLE_FORMULA else "false"
        ]
        
        # Thiết lập nguồn tải mô hình là modelscope
        env = os.environ.copy()
        env["MINERU_MODEL_SOURCE"] = "modelscope"
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Ghi log stdout và stderr ra file trên đĩa để tránh deadlock rò rỉ pipe trên Windows
        stdout_file = temp_out_dir / "mineru_stdout.log"
        stderr_file = temp_out_dir / "mineru_stderr.log"
        
        # Chạy lệnh trong thư mục tạm thời
        with open(stdout_file, "w", encoding="utf-8") as out_f, open(stderr_file, "w", encoding="utf-8") as err_f:
            subprocess.run(
                cmd,
                cwd=str(temp_out_dir),
                env=env,
                stdout=out_f,
                stderr=err_f,
                check=True,
                timeout=dynamic_timeout  # Áp dụng timeout động
            )
        
        # MinerU sinh kết quả ở: temp_out_dir / pdf_stem / auto / pdf_stem.md
        # Hoặc temp_out_dir / pdf_stem / pdf_stem.md
        generated_md = temp_out_dir / pdf_stem / "auto" / f"{pdf_stem}.md"
        if not generated_md.exists():
            generated_md = temp_out_dir / pdf_stem / f"{pdf_stem}.md"
            
        if generated_md.exists():
            return generated_md
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi trích xuất {pdf_path.name} (mã lỗi exit code: {e.returncode})")
        err_msg = ""
        stderr_file = temp_out_dir / "mineru_stderr.log"
        stdout_file = temp_out_dir / "mineru_stdout.log"
        
        if stderr_file.exists():
            with open(stderr_file, "r", encoding="utf-8", errors="replace") as f:
                stderr_content = f.read().strip()
            if stderr_content:
                err_msg = f"\n[Chi tiết lỗi từ mineru cho {pdf_path.name}]:\n{stderr_content}\n"
        
        if not err_msg and stdout_file.exists():
            with open(stdout_file, "r", encoding="utf-8", errors="replace") as f:
                stdout_content = f.read().strip()
            if stdout_content:
                err_msg = f"\n[Chi tiết lỗi]: Không có thông tin stderr từ CLI. Stdout:\n{stdout_content}\n"
                
        tqdm.write(err_msg)
    except subprocess.TimeoutExpired as e:
        logger.error(f"Quá thời gian trích xuất {pdf_path.name} (quá 5 phút). Tự động bỏ qua tệp này.")
    except Exception as e:
        logger.error(f"Lỗi không xác định khi trích xuất {pdf_path.name}: {e}")
    return None


def main():
    ensure_dirs()
    
    # Tạo thư mục tạm thời cho đầu ra thô của MinerU
    temp_out_dir = PROJECT_ROOT / "data" / "processed" / "mineru_temp_raw"
    temp_out_dir.mkdir(parents=True, exist_ok=True)

    if not check_mineru_cli():
        print("\n[!] LỖI: Không tìm thấy công cụ CLI 'mineru' trong môi trường hiện tại.")
        print("Vui lòng kích hoạt môi trường ảo chứa MinerU trước khi chạy (ví dụ: conda activate mineru hoặc kích hoạt venv).")
        sys.exit(1)

    pdf_files = sorted(list(PDF_DIR.glob("*.pdf")))
    print(f"\n=== MinerU Batch Extractor ===")
    print(f"Tìm thấy: {len(pdf_files)} tệp PDF trong thư mục tài liệu.")
    print(f"Đầu ra Markdown sẽ được lưu vào: {MINERU_MD_DIR}\n")

    success_count = 0
    skipped_count = 0

    for pdf_path in tqdm(pdf_files, desc="Trích xuất tài liệu"):
        pdf_stem = pdf_path.stem
        target_md_path = MINERU_MD_DIR / f"{pdf_stem}.md"

        # Nếu đã có file md trích xuất từ trước, bỏ qua để tiết kiệm thời gian
        if target_md_path.exists():
            skipped_count += 1
            continue

        # Chạy trích xuất tài liệu
        generated_md = extract_single_pdf(pdf_path, temp_out_dir)
        
        if generated_md:
            # Sao chép file md vào thư mục phân tách mineru_markdown
            shutil.copy2(generated_md, target_md_path)
            
            # (Tùy chọn) Sao chép cả thư mục ảnh nếu có
            src_images_dir = generated_md.parent / "images"
            if src_images_dir.exists():
                dest_images_dir = MINERU_MD_DIR / "images"
                dest_images_dir.mkdir(parents=True, exist_ok=True)
                for img_file in src_images_dir.glob("*"):
                    shutil.copy2(img_file, dest_images_dir / img_file.name)
            
            success_count += 1
        else:
            logger.warning(f"Thất bại khi trích xuất file: {pdf_path.name}")

    # Diệt các tiến trình fast_api bị treo của MinerU một lần cuối để giải phóng hoàn toàn các file log
    subprocess.run("taskkill /F /FI \"COMMANDLINE eq *mineru.cli.fast_api*\" /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Dọn dẹp thư mục tạm thời thô sau khi hoàn thành (chống lỗi PermissionError trên Windows)
    if temp_out_dir.exists():
        try:
            shutil.rmtree(temp_out_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Không thể xóa hoàn toàn thư mục tạm {temp_out_dir}: {e}")

    print(f"\n==================================================")
    print(f"  HOÀN THÀNH QUY TRÌNH TRÍCH XUẤT BẰNG MINERU")
    print(f"==================================================")
    print(f"- Tổng số PDF tìm thấy : {len(pdf_files)}")
    print(f"- Số file bỏ qua (đã có): {skipped_count}")
    print(f"- Số file trích xuất mới: {success_count}")
    print(f"- Thư mục kết quả      : {MINERU_MD_DIR}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
