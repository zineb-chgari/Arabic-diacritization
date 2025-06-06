import React from 'react';
import {motion} from "framer-motion";
import 'bootstrap/dist/css/bootstrap.min.css';
function NavBar() {
    
    return <nav className="navbar navbar-expand-lg bg-light fixed-top">
        <div className="container-fluid">
            <motion.div ></motion.div>
               
            <button className="navbar-toggler" type="button" data-bs-toggle="collapse"
                    data-bs-target="#navbarTogglerDemo02" aria-controls="navbarTogglerDemo02" aria-expanded="false"
                    aria-label="Toggle navigation">
                <span className="navbar-toggler-icon"></span>
            </button>
            <div className="collapse navbar-collapse" id="navbarTogglerDemo02">
                <ul className="navbar-nav me-auto mb-2 mb-lg-0">
                    <li className="nav-item">
                        <a className="py-2 d-none  nav-link d-md-inline-block" href="#luffy">GitHub</a>
                    </li>
                    
                    <li className="nav-item">
                        <span
      className="position-absolute"
      style={{
        top: '10px',
        right: '20px',
        fontFamily: 'Cairo, sans-serif',
        fontSize: '1.5rem',
        direction: 'rtl',
        color: '#cc5500',
      }}
    >
      تَشْكِيلٌ
    </span>
                        <a className="py-2 d-none nav-link d-md-inline-block" href="#about"></a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
}

export default NavBar;