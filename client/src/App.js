import logo from './logo.svg';
import './App.css';
import MainView from './view/mainView';
import {useState, useEffect} from 'react';

function App() {
  const [methods, setMethods] = useState(null);
  const [datasets, setDatasets] = useState(null);

  useEffect(() => {
    fetch("http://localhost:1234/").then(res => {
      if (res.status === 200) {
        res.json().then(data => {
          setMethods(data.methods);
          setDatasets(data.datasets);
        });
      }
       
    });
  }, []);

  return (
    <div className="App">
      <MainView methods={methods} datasets={datasets} />
    </div>
  );
}

export default App;
