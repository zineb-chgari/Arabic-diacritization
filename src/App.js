// App.jsx
import './App.css';
import NavBar from './diacritizer/nav';
import Main from './diacritizer/main.jsx';
import { useState } from 'react';

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  
  return (
    <div className="App">
      <div className="Content">
        <NavBar />
        <Main
          inputText={inputText}
          setInputText={setInputText}
          outputText={outputText}
          setOutputText={setOutputText}
          
        />
      </div>
    </div>
  );
}

export default App;
