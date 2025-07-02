import prefabricationField
import autograd.numpy as np
import os

def generate_r0s():
    physics = prefabricationField.Physics()
    r0s=np.linspace(0, 30-0.001, 100)
    data=[]

    for r0 in r0s:
        rs=np.linspace(r0, 30, 200)
        for r in rs:
            if r==r0:
                theta=np.pi/2
            else:
                theta=physics.theta_stag(r0, np.pi/2, r)
            data.append([r, theta, r0])
    
    data=np.array(data)
    output_path = os.path.join(os.path.dirname(__file__), 'r0.npy')
    np.save(output_path, data)
    print(f"Generated r0 data saved to {output_path}")

generate_r0s()