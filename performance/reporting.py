import pandas as pd
import logging
from models.clustering import (
    DBSCANModel,
    GMMModel,
    KMeansModel,
)


class Reporting:
    def __init__(self, trades, balance_history, log_file="trading_report.log"):
        """
        Initialize the Reporting module.

        :param trades: DataFrame containing trade details.
        :param balance_history: DataFrame containing balance and performance history.
        :param log_file: Path to the log file.
        """
        self.trades = trades
        self.balance_history = balance_history
        self.log_file = log_file
        self.clustering_models = {
            "DBSCAN": DBSCANModel(),
            "GMM": GMMModel(),
            "KMeans": KMeansModel(),
        }

        # Configure logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("Reporting")
        self.logger.info("Reporting module initialized.")

    def log_event(self, message, level="info"):
        """
        Log a significant event or metric.

        :param message: Event message.
        :param level: Log level ('info', 'warning', 'error').
        """
        levels = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
        self.logger.log(levels.get(level, logging.INFO), message)

    def apply_clustering_models(self, features):
        """
        Apply clustering models to analyze balance history or trade performance.

        :param features: DataFrame or array containing features to cluster.
        :return: Dictionary of clustering results with cluster labels.
        """
        clustering_results = {}
        for name, model in self.clustering_models.items():
            try:
                labels = model.cluster(features)
                clustering_results[name] = labels
                self.logger.info(f"Clustering results for {name}: {set(labels)}")
            except Exception as e:
                self.logger.error(f"Error applying {name}: {e}")
                clustering_results[name] = None
        return clustering_results

    def generate_trade_summary(self):
        """
        Generate a summary of trading activity.

        :return: DataFrame summarizing trading activity.
        """
        try:
            winning_trades = self.trades[self.trades['balance'].diff() > 0]
            losing_trades = self.trades[self.trades['balance'].diff() < 0]
            total_trades = len(self.trades)

            summary = {
                "Total Trades": total_trades,
                "Winning Trades": len(winning_trades),
                "Losing Trades": len(losing_trades),
                "Win Ratio (%)": (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0,
                "Net Profit": self.trades['balance'].iloc[-1] - self.trades['balance'].iloc[0],
                "Average Trade Profit": self.trades['balance'].diff().mean(),
                "Profit Factor": winning_trades['balance'].diff().sum() / abs(losing_trades['balance'].diff().sum()) if len(losing_trades) > 0 else float('inf'),
            }
            self.logger.info("Trade summary generated.")

            # Apply clustering to trade data
            clustering_features = self.trades[['balance', 'timestamp']].dropna().reset_index(drop=True)
            clustering_results = self.apply_clustering_models(clustering_features)

            # Add clustering results to summary
            summary["Clustering Insights"] = clustering_results

            return pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
        except Exception as e:
            self.logger.error(f"Error generating trade summary: {e}")
            raise

    def generate_performance_table(self):
        """
        Generate historical performance tables for daily, weekly, and monthly updates.

        :return: Dictionary containing DataFrames for daily, weekly, and monthly performance.
        """
        try:
            self.logger.info("Generating historical performance tables.")

            # Set index as datetime for resampling
            self.balance_history['timestamp'] = pd.to_datetime(self.balance_history['timestamp'])
            self.balance_history.set_index('timestamp', inplace=True)

            # Daily Performance
            daily = self.balance_history['balance'].resample('D').last()
            daily_perf = pd.DataFrame({
                "Date": daily.index,
                "Balance": daily.values,
                "Daily Change (%)": daily.pct_change() * 100,
            }).dropna()

            # Weekly Performance
            weekly = self.balance_history['balance'].resample('W').last()
            weekly_perf = pd.DataFrame({
                "Week": weekly.index,
                "Balance": weekly.values,
                "Weekly Change (%)": weekly.pct_change() * 100,
            }).dropna()

            # Monthly Performance
            monthly = self.balance_history['balance'].resample('M').last()
            monthly_perf = pd.DataFrame({
                "Month": monthly.index,
                "Balance": monthly.values,
                "Monthly Change (%)": monthly.pct_change() * 100,
            }).dropna()

            # Apply clustering to performance data
            clustering_features = daily_perf[['Balance', 'Daily Change (%)']].dropna().reset_index(drop=True)
            clustering_results = self.apply_clustering_models(clustering_features)
            self.logger.info(f"Performance clustering insights: {clustering_results}")

            self.logger.info("Historical performance tables created.")
            return {"daily": daily_perf, "weekly": weekly_perf, "monthly": monthly_perf}
        except Exception as e:
            self.logger.error(f"Error generating performance tables: {e}")
            raise

    def export_report(self, file_path="performance_report.xlsx"):
        """
        Export a comprehensive report as an Excel file.

        :param file_path: Path to the Excel file.
        :return: None.
        """
        try:
            self.logger.info(f"Exporting report to {file_path}.")
            with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
                # Trade Summary
                self.generate_trade_summary().to_excel(writer, sheet_name="Trade Summary", index=False)

                # Historical Performance
                performance_tables = self.generate_performance_table()
                performance_tables['daily'].to_excel(writer, sheet_name="Daily Performance", index=False)
                performance_tables['weekly'].to_excel(writer, sheet_name="Weekly Performance", index=False)
                performance_tables['monthly'].to_excel(writer, sheet_name="Monthly Performance", index=False)

            self.logger.info(f"Report successfully exported to {file_path}.")
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            raise

