# Power BI Templates and Resources

This folder contains templates and resources for Power BI dashboard creation.

## 📁 Folder Structure

```
powerbi/
├── POWER_BI_DASHBOARD_GUIDE.md    # Complete setup guide
├── data/                          # Exported CSV files for Power BI
│   ├── transactions_processed.csv
│   ├── credit_data_processed.csv
│   ├── monthly_summary.csv
│   ├── customer_segments.csv
│   ├── fraud_analysis.csv
│   ├── credit_risk_summary.csv
│   └── data_dictionary.csv
├── screenshots/                   # Documentation screenshots
│   ├── setup/
│   ├── data_import/
│   ├── modeling/
│   ├── visualizations/
│   ├── layout/
│   ├── publishing/
│   └── final_dashboard/
└── templates/                     # Power BI templates and examples
    ├── DAX_formulas.txt
    ├── color_schemes.json
    └── README.md (this file)
```

## 🎯 Quick Start

1. **Export Data**: Run the data export script first

   ```bash
   python utils/export_for_powerbi.py
   ```

2. **Open Power BI Desktop**: Install if not already available

3. **Follow the Guide**: Use POWER_BI_DASHBOARD_GUIDE.md for step-by-step instructions

4. **Import Data**: Connect to CSV files in the data/ folder

5. **Build Dashboard**: Follow visualization guidelines in the main guide

## 📊 Dashboard Components

### Key Performance Indicators (KPIs)

- Total Transaction Volume
- Fraud Detection Rate
- Active Customers
- Average Transaction Value

### Visualizations

- Time Series Charts (Monthly trends)
- Distribution Charts (Transaction amounts, Risk categories)
- Heatmaps (Transaction patterns by time)
- Scatter Plots (Credit risk analysis)
- Geographic Maps (if location data available)

### Interactive Features

- Date range slicers
- Transaction type filters
- Customer segment filters
- Risk category filters

## 🎨 Design Guidelines

### Color Scheme

- **Primary**: Dark Blue (#1f4e79)
- **Secondary**: Orange (#ff6b35)
- **Success**: Green (#28a745)
- **Warning**: Yellow (#ffc107)
- **Danger**: Red (#dc3545)

### Typography

- **Headers**: Segoe UI Bold, 16-18pt
- **Body**: Segoe UI Regular, 10-12pt
- **KPIs**: Segoe UI Bold, 24-28pt

## 📋 Data Dictionary

Refer to `data/data_dictionary.csv` for complete column descriptions and data types.

## 🔧 Troubleshooting

### Common Issues

1. **Data Import Errors**: Check CSV encoding (should be UTF-8)
2. **Date Format Issues**: Ensure dates are in YYYY-MM-DD format
3. **Performance Issues**: Limit data volume or use aggregated tables
4. **Relationship Errors**: Verify customer_id consistency across tables

### Support Resources

- [Power BI Documentation](https://docs.microsoft.com/power-bi/)
- [DAX Function Reference](https://docs.microsoft.com/dax/)
- [Power BI Community](https://community.powerbi.com/)

---

**Created by**: Financial Analytics Team  
**Last Updated**: September 2025
