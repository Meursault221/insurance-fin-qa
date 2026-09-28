"""company 配置加载与文件名推断测试。"""

import pytest

from src import config


class TestLoadCompanies:
    def test_加载10家公司(self):
        meta, profiles = config.load_companies()
        assert len(profiles) == 10
        assert meta.get("report_year") == 2025

    def test_公司名不重复(self):
        _, profiles = config.load_companies()
        names = [p.company for p in profiles]
        assert len(names) == len(set(names))

    def test_每家都有报告名(self):
        _, profiles = config.load_companies()
        for p in profiles:
            assert p.report_name, f"{p.company} 缺 report_name"

    def test_有未核对标记(self):
        """配置里应明确标出哪些股票代码还没核对，避免误用。"""
        _, profiles = config.load_companies()
        assert all(isinstance(p.verified, bool) for p in profiles)

    def test_get_profile(self):
        p = config.get_profile("pingan")
        assert p.company == "中国平安"
        assert "601318.SH" in p.stock_codes

    def test_get_profile不存在时报错(self):
        with pytest.raises(KeyError):
            config.get_profile("不存在的公司")


class TestFilenameGuess:
    @pytest.mark.parametrize("filename,expected", [
        ("pingan_2025_annual.pdf", "pingan"),
        ("china_life_2025_annual.pdf", "china_life"),
        ("cpic_2025_annual.pdf", "cpic"),
        ("new_china_life_2025_annual.pdf", "new_china_life"),
        ("picc_pc_2025_annual.pdf", "picc_pc"),
        ("中国平安2025年年度报告.pdf", "pingan"),
    ])
    def test_能认出文件名(self, filename, expected):
        assert config.guess_key_from_filename(filename) == expected

    def test_认不出来返回None不瞎猜(self):
        assert config.guess_key_from_filename("random_report_2025.pdf") is None

    def test_提取年份(self):
        assert config.guess_year_from_filename("pingan_2024_annual.pdf") == 2024
        assert config.guess_year_from_filename("no_year.pdf", 2025) == 2025

    def test_识别报告类型(self):
        assert config.guess_report_type_from_filename("x_2025_semi.pdf") == "semi_annual"
        assert config.guess_report_type_from_filename("x_2025_annual.pdf") == "annual"
        assert config.guess_report_type_from_filename("x_2025_半年报.pdf") == "semi_annual"

    def test_报告名生成(self):
        assert config.guess_report_name(2025, "annual") == "2025年年度报告"
        assert config.guess_report_name(2025, "annual", is_hk=True) == "2025年報"


class TestBatchDedup:
    """Windows 上 glob 大小写不敏感，"*.pdf" 和 "*.PDF" 会命中同一批文件。"""

    def test_同一文件不被处理两次(self, tmp_path):
        import shutil
        from pathlib import Path

        from src import pipeline

        fixture = Path(__file__).parent / "fixtures" / "demo_company_2025_annual.pdf"
        if not fixture.exists():
            import pytest
            pytest.skip("合成样本未生成")

        shutil.copy(fixture, tmp_path / "pingan_2025_annual.pdf")
        shutil.copy(fixture, tmp_path / "cpic_2025_annual.pdf")

        results = pipeline.process_batch(tmp_path, write=False)
        names = [r.doc.meta.source_file for r in results]
        assert sorted(names) == ["cpic_2025_annual.pdf", "pingan_2025_annual.pdf"]

    def test_认不出公司名的被跳过而非瞎猜(self, tmp_path):
        import shutil
        from pathlib import Path

        from src import pipeline

        fixture = Path(__file__).parent / "fixtures" / "demo_company_2025_annual.pdf"
        if not fixture.exists():
            import pytest
            pytest.skip("合成样本未生成")

        shutil.copy(fixture, tmp_path / "完全无关的名字.pdf")
        results = pipeline.process_batch(tmp_path, write=False)
        assert results == [], "文件名认不出来时不应该硬处理"


class TestPaths:
    def test_目录都在项目内(self):
        for d in (config.RAW_DIR, config.EXTRACTED_DIR, config.CHUNKS_DIR,
                  config.INDEX_DIR):
            assert config.PROJECT_ROOT in d.parents or d.parent == config.PROJECT_ROOT
