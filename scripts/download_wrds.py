from pathlib import Path

import wrds


def download_wti(start="2021-01-01", end="2025-12-31"):
    db = wrds.Connection()

    try:
        df = db.raw_sql(
            """
            SELECT
                p.date_,
                p.futcode,
                c.dsmnem,
                c.contrdate,
                c.lasttrddate,
                p.settlement,
                p.volume,
                p.openinterest
            FROM tr_ds_fut.dsfutcontrval AS p
            JOIN tr_ds_fut.wrds_contract_info AS c
                ON p.futcode = c.futcode
            WHERE c.contrcode = %(contrcode)s
                AND p.date_ BETWEEN %(start)s AND %(end)s
            ORDER BY p.date_, c.lasttrddate
            """,
            params={
                "contrcode": 1986,
                "start": start,
                "end": end,
            },
        )
    finally:
        db.close()

    path = Path("data/raw/wti_2021_2025.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=False)

    print(f"Downloaded {len(df):,} observations.")
    print(f"Unique contracts: {df['futcode'].nunique()}")
    print(f"Saved to {path}")

    return df


if __name__ == "__main__":
    download_wti()