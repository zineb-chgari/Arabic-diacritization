
import AutosaveTextarea from "./textarea";
import AutosaveTextareanondiacr from "./non_diacritizedarea";
import Button from "./button";

export default function Main({ inputText, setInputText, outputText, setOutputText }) {
   // Résultat diacrité

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', padding: '20px', maxWidth: '100%', margin: '0 auto' }}>
      <div style={{ display: 'flex', gap: '20px', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ flex: 1 }}>
          <AutosaveTextarea text={outputText} />
        </div>
        <div style={{ flex: 1 }}>
          <AutosaveTextareanondiacr text={inputText} setText={setInputText} />
        </div>
      </div>
      <div style={{ alignSelf: 'center' }}>
        <Button inputText={inputText} setOutputText={setOutputText} />
      </div>
    </div>
  );
}