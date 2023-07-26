import axios from "axios";
import { useState } from "react";

type FormProps = {
  onFormSubmit: (data: object) => void;
};

export default function Form({ onFormSubmit }: FormProps) {
  const [man, setMan] = useState("");
  const [age, setAge] = useState("");
  const [durationOfDiabetes, setDurationOfDiabetes] = useState("");
  const [baseHbA1cLevel, setBaseHbA1cLevel] = useState("");
  const [objHbA1cLevel, setObjHbA1cLevel] = useState("");
  const [hypoRate, setHypoRate] = useState("");
  const [hypoRateRR, setHypoRateRR] = useState("");
  const [annualCost, setAnnualCost] = useState("");
  const [manifestations, setManifestations] = useState<string[]>([]);

  const handleManifestationsChange = (event: {
    target: { value: string; checked: boolean };
  }) => {
    const value = event.target.value;
    if (event.target.checked) {
      setManifestations([...manifestations, value]);
    } else {
      setManifestations(manifestations.filter((m) => m !== value));
    }
  };

  const handleSubmit = (event: { preventDefault: () => void }) => {
    event.preventDefault();
    // Aquí se puede agregar la lógica para enviar los datos del formulario a un servidor o realizar alguna acción adicional
    let formData = {
      man,
      age,
      durationOfDiabetes,
      hypoRate,
      hypoRateRR,
      baseHbA1cLevel,
      objHbA1cLevel,
      annualCost,
      manifestations,
    };
    console.log(formData)

    // POST request
    axios.post('http://127.0.0.1:5000/diabetes', JSON.stringify(formData), {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    }).then((response) => {
        onFormSubmit(response.data);
      })
  };


  return (
    <form onSubmit={handleSubmit} className="max-w-8xl my-auto">
      <div className="grid grid-cols-3 gap-4">
        <div className="mb-4">
          <label htmlFor="man" className="block mb-2 font-bold" style={{ color: '#1D3557'}}>
            Sex
          </label>
          <select
            id="man"
            name="man"
            value={man}
            onChange={(event) => setMan(event.target.value)}
            className="p-2 w-full"
            style={{ backgroundColor: '#F1FAEE', color: '#6b7280' }}
            required
          >
            <option value="">Select an option</option>
            <option value="true">Male</option>
            <option value="false">Female</option>
          </select>
        </div>
        <div className="mb-4">
          <label htmlFor="age" className="block mb-2 font-bold" style={{ color: '#1D3557'}}>
            Age
          </label>
          <input
            type="number"
            id="age"
            name="age"
            value={age}
            placeholder="Introduzca un número"
            onChange={(event) => setAge(event.target.value)}
            className="p-2 w-full"
            style={{ backgroundColor: '#F1FAEE' }}
            required
          />
        </div>
        <div className="mb-4">
          <label
            htmlFor="durationOfDiabetes"
            className="block mb-2 font-bold"
            style={{ color: '#1D3557'}}
          >
           Duration of diabetes (in years)
          </label>
          <input
            type="number"
            id="durationOfDiabetes"
            name="durationOfDiabetes"
            value={durationOfDiabetes}
            placeholder="Introduzca un número"
            onChange={(event) => setDurationOfDiabetes(event.target.value)}
            className="p-2 w-full"
            style={{ backgroundColor: '#F1FAEE' }}
            required
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="mb-4">
          <label
            htmlFor="baseHbA1cLevel"
            className="block mb-2 font-bold"
            style={{ color: '#1D3557'}}
          >
           Base HbA1c level (%)
          </label>
          <input
            type="number"
            id="baseHbA1cLevel"
            name="baseHbA1cLevel"
            value={baseHbA1cLevel}
            placeholder="Introduzca un número"
            onChange={(event) => setBaseHbA1cLevel(event.target.value)}
            className="p-2 w-full"
            style={{ backgroundColor: '#F1FAEE' }}
            required
          />
        </div>
        <div className="mb-4">
          <label
            htmlFor="objHbA1cLevel"
            className="block mb-2 font-bold"
            style={{ color: '#1D3557'}}
          >
            HbA1c Target Level (%)
          </label>
          <input
            type="number"
            id="objHbA1cLevel"
            name="objHbA1cLevel"
            value={objHbA1cLevel}
            placeholder="Introduzca un número"
            onChange={(event) => setObjHbA1cLevel(event.target.value)}
            className="p-2 w-full"
            style={{ backgroundColor: '#F1FAEE' }}
            required
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="mb-4">
          <label
            htmlFor="hypoRate"
            className="block mb-2 font-bold"
            style={{ color: '#1D3557'}}
          >
            Severe hypoglycemic episodes (per year)
          </label>
          <input
            type="number"
            id="hypoRate"
            name="hypoRate"
            value={hypoRate}
            placeholder="Introduzca un número"
            onChange={(event) => setHypoRate(event.target.value)}
            className="p-2 w-full"
            style={{ backgroundColor: '#F1FAEE' }}
            required
          />
        </div>
        <div className="mb-4">
          <label
            htmlFor="hypoRateRR"
            className="block mb-2 font-bold"
            style={{ color: '#1D3557'}}
          >
            Target number of severe hypoglycemic episodes (per year)
          </label>
          <input
            type="number"
            id="hypoRateRR"
            name="hypoRateRR"
            placeholder="Introduzca un número"
            value={hypoRateRR}
            onChange={(event) => setHypoRateRR(event.target.value)}
            className="p-2 w-full"
            style={{ backgroundColor: '#F1FAEE' }}
            required
          />
        </div>
      </div>
      <div className="mb-4">
        <label
          htmlFor="annualCost"
          className="block mb-2 font-bold"
          style={{ color: '#1D3557'}}
        >
          Treatment cost (in dollars)
        </label>
        <input
          type="number"
          id="annualCost"
          name="annualCost"
          value={annualCost}
          placeholder="Introduzca un número"
          onChange={(event) => setAnnualCost(event.target.value)}
          style={{ backgroundColor: '#F1FAEE' }}
          className="p-2 w-full"
          required
        />
      </div>
      <div className="mb-4">
        <label
          htmlFor="manifestations"
          className="block mb-2 font-bold"
          style={{ color: '#1D3557'}}
        >
          Clinical manifestations
        </label>
        <div className="grid grid-cols-2 gap-4">
          <label
            htmlFor="Retinopatía de Fondo"
            className="inline-flex items-center mt-2"
          >
            <input
              id="BGRET"
              name="manifestations"
              type="checkbox"
              value="BGRET"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("BGRET")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}
            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Background Retinopathy</span>
          </label>
          <label
            htmlFor="Retinopatía Proliferativa"
            className="inline-flex items-center mt-2"
          >
            <input
              id="PRET"
              name="manifestations"
              type="checkbox"
              value="PRET"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("PRET")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}
            />
            <span className="ml-2" style={{ color: '#1D3557'}}>
            Proliferative Retinopathy
            </span>
          </label>
          <label
            htmlFor="Edema Macular Diabético"
            className="inline-flex items-center mt-2"
          >
            <input
              id="ME"
              name="manifestations"
              type="checkbox"
              value="ME"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ME")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}
            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Macular Edema</span>
          </label>
          <label
            htmlFor="Enfermedad Renal Terminal"
            className="inline-flex items-center mt-2"
          >
            <input
              id="ESRD"
              name="manifestations"
              type="checkbox"
              value="ESRD"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ESRD")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}
            />
            <span className="ml-2" style={{ color: '#1D3557'}}>
            End-Stage Renal Disease
            </span>
          </label>
          <label htmlFor="Ceguera" className="inline-flex items-center mt-2">
            <input
              id="BLI"
              name="manifestations"
              type="checkbox"
              value="BLI"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("BLI")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Blindness</span>
          </label>
          <label
            htmlFor="Microalbuminuria"
            className="inline-flex items-center mt-2"
          >
            <input
              id="ALB1"
              name="manifestations"
              type="checkbox"
              value="ALB1"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ALB1")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Microalbuminuria</span>
          </label>
          <label
            htmlFor="Macroalbuminuria"
            className="inline-flex items-center mt-2"
          >
            <input
              id="ALB2"
              name="manifestations"
              type="checkbox"
              value="ALB2"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ALB2")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Macroalbuminuria</span>
          </label>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <label htmlFor="Angina" className="inline-flex items-center mt-2">
            <input
              id="ANGINA"
              name="manifestations"
              type="checkbox"
              value="ANGINA"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("ANGINA")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Angina</span>
          </label>
          <label htmlFor="Ictus" className="inline-flex items-center mt-2">
            <input
              id="STROKE"
              name="manifestations"
              type="checkbox"
              value="STROKE"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("STROKE")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Stroke</span>
          </label>
          <label
            htmlFor="Infarto de Miocardio"
            className="inline-flex items-center mt-2"
          >
            <input
              id="MI"
              name="manifestations"
              type="checkbox"
              value="MI"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("MI")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Myocardial Infarction</span>
          </label>
          <label
            htmlFor="Fallo Cardíaco"
            className="inline-flex items-center mt-2"
          >
            <input
              id="HF"
              name="manifestations"
              type="checkbox"
              value="HF"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("HF")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Heart Failure</span>
          </label>
          <label htmlFor="Neuropatía" className="inline-flex items-center mt-2">
            <input
              id="NEU"
              name="manifestations"
              type="checkbox"
              value="NEU"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("NEU")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>Neuropathy</span>
          </label>
          <label
            htmlFor="Amputación de Extremidades Inferiores"
            className="inline-flex items-center mt-2"
          >
            <input
              id="LEA"
              name="manifestations"
              type="checkbox"
              value="LEA"
              onChange={handleManifestationsChange}
              checked={manifestations.includes("LEA")}
              className="form-checkbox h-5 w-5"
              style={{ backgroundColor: '#457B9D' }}

            />
            <span className="ml-2" style={{ color: '#1D3557'}}>
              Lower Extremity Amputation
            </span>
          </label>
        </div>
      </div>
      <div className="mt-8 flex flex-col items-center">
        <button
          type="submit"
          className="font-bold py-2 px-4 rounded"
          style={{
            backgroundColor: "#E63946", // Cambia el color de fondo del botón
            color: "#F1FAEE", // Cambia el color del texto del botón
          }}
        >
          Enviar
        </button>
      </div>
    </form>
  );
}
