import React, { useState } from "react";
import axios from "axios";

function App() {
  const [rows, setRows] = useState([]);
  const [columns, setColumns] = useState([]);
  const [layoffs, setLayoffs] = useState(1);
  const [results, setResults] = useState([]);

  const uploadCSV = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    const res = await axios.post("http://localhost:8000/upload", formData);
    setColumns(res.data.columns);
    setRows(res.data.data);
    setResults([]);
  };

  const handleEdit = (i, col, value) => {
    const updated = [...rows];
    updated[i][col] = value;
    setRows(updated);
  };

  const predictLayoffs = async () => {
    const res = await axios.post("http://localhost:8000/predict", rows, {
      headers: { "Content-Type": "application/json" },
      params: { layoffs },
    });
    setResults(res.data);
  };

  return (
    <div className="page">
      <div className="container">
        <h1 className="title">Layoff Risk Analyzer</h1>
        <input type="file" onChange={uploadCSV} className="file-input" />

        {rows.length > 0 && (
          <>
            <div className="table-scroll">
              <table className="styled-table">
                <thead>
                  <tr>
                    {columns.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => (
                    <tr key={i}>
                      {columns.map((col) => (
                        <td key={col}>
                          <input
                            value={row[col]}
                            onChange={(e) => handleEdit(i, col, e.target.value)}
                            className="cell-input"
                          />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="controls">
              <input
                type="number"
                value={layoffs}
                onChange={(e) => setLayoffs(e.target.value)}
                className="number-input"
                placeholder="Number to lay off"
              />
              <button onClick={predictLayoffs} className="submit-btn">
                Submit
              </button>
            </div>
          </>
        )}

        {results.length > 0 && (
          <div className="results">
            <h2 className="subtitle">Least Efficient Employees</h2>
            <div className="table-scroll">
              <table className="styled-table">
                <thead>
                  <tr>
                    {Object.keys(results[0]).map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((val, j) => (
                        <td key={j}>{val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
