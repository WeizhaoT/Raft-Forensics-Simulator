# Raft-Forensics-Simulator

## Simulator



## Dashboard

1. Install Grafana and get the service running following the [tutorial](https://grafana.com/docs/grafana/latest/?utm_source=grafana_footer)
2. Open your web browser and go to http://localhost:3000/, import dashboard from file `Raft Forensics.json`
3. Install mysql ([documentation](https://dev.mysql.com/doc/refman/8.0/en/installing.html))
4. Add a user `test` with passwoard `test`, create a database named `forensics` (the configuration is specified in `sql.py`)
5. Run `dashboard.py` after starting `simulate.py`,  example:

    ```
    python simulate.py -x 100 -e 20 -w 0.5 -T 50 --depth 0.1 -d --bvote -s 14 -p ./data
    python dashboard.py --exp data/badvote-100-20-0-0.10-14
    ```
