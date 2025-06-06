import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export default function AutosaveTextareanondiacr({ text, setText }) {

  
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);
  const [charCount, setCharCount] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (text && text.trim() !== '') {
        setIsSaving(true);
        
        // Simulation d'appel API
        setTimeout(() => {
          console.log('Texte sauvegardé:', text);
          setIsSaving(false);
          setLastSaved(new Date().toLocaleTimeString());
          setCharCount(text.length);
        }, 1000);
      }
    }, 2000);

    return () => clearTimeout(timer);
  }, [text]);

  return (
    <div style={{
      maxWidth: '1000px',
      margin: '8rem auto 0', 
      marginRight: '10px',
      padding: '1.5rem',
      backgroundColor: 'white',
      borderRadius: '10px',
      boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)'
    }}>
    <button
  style={{
    fontSize: '1.2rem',
    padding: '10px 40px',
    marginLeft:'100px',
    backgroundColor: '#cc5500',
    color: 'white',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    direction: 'rtl',
  }}
  lang="ar"
>
النص غير مشكول
</button>

      
      
<motion.textarea
  initial={{
    x: '700px',
    borderColor: 'white'
  }}
  animate={{
    x: '0',
    borderColor: 'white'
  }}
  transition={{
    duration: 2,
    delay: 1
  }}
  value={text}
  onChange={(e) => setText(e.target.value)}
  rows={6}
  lang="ar"
  dir="rtl"
  placeholder="اكتب نصك غير مشكول هنا"
  style={{
    width: '100%',
    padding: '12px',
    border: '5px solid white',
    borderRadius: '6px',
    fontSize: '1rem',
    resize: 'vertical',
    minHeight: '150px',
    transition: 'all 0.3s ease',
    outline: 'none',
    marginBottom: '0.5rem'
  }}
  whileFocus={{
    borderWidth: "0px"
  }}
/>
      
      <div style={{
        marginTop: '0.5rem',
        fontSize: '0.9rem',
        color: '#666',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center' // Alignement vertical au centre
      }}>
        {isSaving ? (
          <motion.div 
            style={{
              color: '#3a86ff',
              display: 'flex',
              alignItems: 'center'
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ repeat: Infinity, duration: 1 }}
          >
            <span style={{
              marginRight: '6px',
              animation: 'pulse 1s infinite'
            }}>•</span>
            يَجْرِي الحِفْظ...
          </motion.div>
        ) : (
          lastSaved && (
            <div style={{
              display: 'flex',
              gap: '1rem',
              width: '100%',
              justifyContent: 'space-between'
            }}>
              <span>آخِرُ حِفْظٍ {lastSaved}</span>
              <span style={{ color: '#666', fontStyle: 'italic' }}>
                {charCount} حرف
              </span>
            </div>
          )
        )}
      </div>

      
    </div>
    
  );
}